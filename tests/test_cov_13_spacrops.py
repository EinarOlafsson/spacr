"""What the stitcher does when the file, the detector or the GPU misbehaves.

Stitching runs unattended over thousands of tiles, so every guard here exists
because the alternative is a silently wrong mosaic rather than a crash: a TIFF
that reports no axes, a detector that ignores the keypoint budget, a RANSAC
call that returns no inlier mask, a CUDA build that is not there. Each has to
produce a defined, checkable result, and the two stitcher classes have to
agree with each other about what that result is -- they carry the same reader
twice, and a fix applied to one of them only is the failure this file is
shaped to catch.
"""
from __future__ import annotations

import csv
import os
import types

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import pytest
import tifffile

from spacr import spacrops
from spacr.spacrops import StitchedMultiAligner, spacrStitcher


# ---------------------------------------------------------------------------
# a TIFF reader that reports no axes metadata
# ---------------------------------------------------------------------------

class _AxelessSeries:
    """A tifffile series stand-in whose ``axes`` attribute is ``None``."""

    def __init__(self, array):
        self.axes = None
        self.shape = array.shape
        self.dtype = array.dtype
        self._array = array

    def asarray(self):
        return self._array


class _AxelessTiffFile:
    """Reads the real pixels, then hides the axes metadata from the caller."""

    def __init__(self, path):
        self._array = np.asarray(tifffile.imread(path))

    @property
    def series(self):
        return [_AxelessSeries(self._array)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def axeless_reader(monkeypatch):
    """Make every TIFF spacrops opens report no axes string."""
    monkeypatch.setattr(spacrops, "tifffile",
                        types.SimpleNamespace(TiffFile=_AxelessTiffFile,
                                              imwrite=tifffile.imwrite,
                                              imread=tifffile.imread))


def _plane(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(h, w), dtype=np.uint16)


def _write_stack(path, planes):
    """Write a plane stack without letting tifffile guess it is RGB."""
    tifffile.imwrite(path, np.stack(planes), photometric="minisblack")
    return path


def _st(tmp_path, **kw):
    kw.setdefault("outdir", str(tmp_path / "out"))
    kw.setdefault("save_qc", False)
    kw.setdefault("save_stitched_default", False)
    kw.setdefault("feature_cache_mode", "ram")
    return spacrStitcher(**kw)


# ---------------------------------------------------------------------------
# reading a TIFF that carries no axes metadata
# ---------------------------------------------------------------------------

def test_a_channel_token_in_the_filename_names_the_leading_axis(tmp_path,
                                                                axeless_reader):
    """With no axes metadata the filename decides C versus Z.

    A Yokogawa export names the channel in the file (``..._c2_...``). Reading
    such a stack as ZYX would max-project the channels together, producing one
    plane that is every stain at once -- which still looks like an image and
    is silently unusable.
    """
    stack = np.stack([_plane(seed=0), _plane(seed=1), _plane(seed=2)])
    path = _write_stack(str(tmp_path / "10X_c1_A1_r01f01.tif"), stack)

    st = _st(tmp_path, mip=True)

    assert np.array_equal(st._read_plane(path, ch=0), stack[0])
    assert np.array_equal(st._read_plane(path, ch=2), stack[2])


def test_without_a_channel_token_a_projected_read_treats_the_stack_as_z(
        tmp_path, axeless_reader):
    """No axes and no channel token: ``mip`` says what the leading axis is.

    A caller who asked for a maximum projection is telling the reader the
    stack is a Z series; a caller who did not is left with the channel reading,
    which is what a plain multi-plane tile from a plate scan is.
    """
    stack = np.stack([_plane(seed=3), _plane(seed=4), _plane(seed=5)])
    path = _write_stack(str(tmp_path / "10X_A1_r01f01.tif"), stack)

    projected = _st(tmp_path, mip=True)._read_plane(path, ch=0)
    per_channel = _st(tmp_path, mip=False)._read_plane(path, ch=1)

    assert np.array_equal(projected, stack.max(axis=0))
    assert np.array_equal(per_channel, stack[1])


def test_the_multi_aligner_reads_an_axesless_stack_the_same_way(tmp_path,
                                                                axeless_reader):
    """The second stitcher carries its own copy of the reader.

    They are separate methods on separate classes, so a rule applied to one is
    not applied to the other unless something says so.
    """
    stack = np.stack([_plane(seed=6), _plane(seed=7)])
    token = _write_stack(str(tmp_path / "10X_c1_A1_r01f01.tif"), stack)
    plain = _write_stack(str(tmp_path / "10X_A1_r01f02.tif"), stack)

    al = StitchedMultiAligner(outdir=str(tmp_path / "ma"), mip=True)

    assert np.array_equal(al._read_plane(token, ch=1), stack[1])
    assert np.array_equal(al._read_plane(plain, ch=0), stack.max(axis=0))


def test_the_channel_count_of_an_axesless_stack_is_guessed_from_its_shape(
        tmp_path, axeless_reader):
    """Without an axes string the shape is the only evidence there is.

    The count decides how many planes the mosaic builder asks for, so guessing
    1 for a real three-channel tile drops two stains from the output.
    """
    three = _write_stack(str(tmp_path / "three.tif"),
                         [_plane(seed=i) for i in range(3)])
    flat = str(tmp_path / "flat.tif")
    tifffile.imwrite(flat, _plane(seed=9))

    assert spacrStitcher._get_channel_count_tif(three) == 3
    assert spacrStitcher._get_channel_count_tif(flat) == 1
    assert StitchedMultiAligner._get_channel_count_tif(three) == 3
    assert StitchedMultiAligner._get_channel_count_tif(flat) == 1


# ---------------------------------------------------------------------------
# an axis label the reader does not know
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [spacrStitcher, StitchedMultiAligner])
def test_an_axis_label_the_reader_does_not_know_is_taken_at_its_first_plane(
        tmp_path, monkeypatch, cls):
    """An unrecognised axis must collapse to plane 0, not raise or vanish.

    The guessed axes string is the only description an axes-less file has, and
    a future label added there -- or one arriving from a caller's ``arr_axes``
    -- must still leave a 2-D plane behind. Falling through with no slicer
    would hand the caller a 3-D array under a name that promises 2-D.
    """
    monkeypatch.setattr(cls, "_guess_axes_from_shape",
                        staticmethod(lambda shape: "QYX"))
    reader = cls(outdir=str(tmp_path / f"out_{cls.__name__}"))
    stack = np.stack([_plane(seed=11), _plane(seed=12)])

    plane = reader._normalize_to_yx(stack, ch=1)

    assert plane.shape == stack.shape[1:]
    assert np.array_equal(plane, stack[0])


# ---------------------------------------------------------------------------
# a detector that ignores the keypoint budget
# ---------------------------------------------------------------------------

def test_a_detector_that_overshoots_the_budget_is_capped_before_matching(
        tmp_path, monkeypatch):
    """The keypoint budget is what keeps a huge well inside memory.

    The detector is asked to respect it, but the budget is enforced again on
    the way out because the descriptors go into the pairwise matcher and into
    the on-disk feature cache: one tile returning ten times the cap is a
    quadratic blow-up in the matcher, not a slightly larger array.
    """
    path = str(tmp_path / "10X_c1_A1_r01f01.tif")
    tifffile.imwrite(path, _plane(seed=13))
    st = _st(tmp_path, max_keypoints=3, downsample=1.0)

    points = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0],
                       [30.0, 30.0], [40.0, 40.0]], dtype=np.float32)
    descriptors = np.arange(5 * 32, dtype=np.uint8).reshape(5, 32)
    monkeypatch.setattr(st, "_detect_and_describe",
                        lambda image: (points, descriptors))

    feature = st._compute_features_one(path, channel_index=0)

    assert feature["pts"].shape == (3, 2)
    assert feature["desc"].shape == (3, 32)
    # The cap keeps the points furthest from the centroid, which is the set
    # that spans the tile rather than a cluster in the middle of it.
    kept = {tuple(p) for p in feature["pts"].tolist()}
    assert kept == {(0.0, 0.0), (40.0, 40.0), (30.0, 30.0)}


# ---------------------------------------------------------------------------
# RANSAC that reports no inlier mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [spacrStitcher, StitchedMultiAligner])
def test_a_transform_with_no_inlier_mask_scores_zero_rather_than_crashing(
        monkeypatch, cls):
    """OpenCV may return a transform and no mask; the score must then be zero.

    The inlier ratio IS the pair score the mosaic thresholds on. Treating a
    missing mask as anything other than "no support" would let an unverified
    transform through the score filter and place a tile by it.
    """
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]], dtype=np.float32)
    monkeypatch.setattr(cv2, "estimateAffinePartial2D",
                        lambda *args, **kwargs: (affine, None))
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                      dtype=np.float32)

    M, mask, ratio = cls._affine_from_pts(points, points, 3.0)

    assert mask is None
    assert ratio == 0.0
    assert np.allclose(M, affine)


# ---------------------------------------------------------------------------
# a machine with no torch
# ---------------------------------------------------------------------------

class _RecordingCellposeModel:
    """Records how the Cellpose model was built; loads no weights."""

    built = []

    def __init__(self, **kwargs):
        type(self).built.append(kwargs)


def _no_torch(monkeypatch):
    """Make ``import torch`` fail the way an install without it does."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_cellpose_outlines_fall_back_to_cpu_when_torch_cannot_be_imported(
        tmp_path, monkeypatch):
    """A missing torch means no GPU, not a failed run.

    QC outlines are a diagnostic drawn beside the stitch. Letting the import
    error escape would take a whole stitching run down over a picture, and
    asking torch whether CUDA is available is the only reason torch is
    touched here at all.
    """
    pytest.importorskip("cellpose")
    from cellpose import models as cp_models
    from spacr import utils as spacr_utils

    _RecordingCellposeModel.built = []
    monkeypatch.setattr(cp_models, "CellposeModel", _RecordingCellposeModel)
    monkeypatch.setattr(spacr_utils, "_resolve_cellpose_pretrained",
                        lambda name: "cpsam")
    st = _st(tmp_path, outline_source="cellpose")
    _no_torch(monkeypatch)

    model = st._get_cellpose_model()

    assert isinstance(model, _RecordingCellposeModel)
    assert _RecordingCellposeModel.built == [{"gpu": False,
                                              "pretrained_model": "cpsam"}]
    # Built once and cached: a fresh model per tile re-loads the weights.
    assert st._get_cellpose_model() is model
    assert len(_RecordingCellposeModel.built) == 1


def test_cellpose_outlines_still_refuse_when_cellpose_itself_is_absent(
        tmp_path, monkeypatch):
    """The torch fallback must not swallow a missing cellpose too.

    Without cellpose there is nothing to fall back to, and a run that
    continued would draw no outlines while reporting that it had.
    """
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "cellpose" or name.startswith("cellpose."):
            raise ImportError("No module named 'cellpose'")
        return real_import(name, *args, **kwargs)

    st = _st(tmp_path, outline_source="cellpose")
    monkeypatch.setattr(builtins, "__import__", refuse)

    with pytest.raises(RuntimeError, match="requires `cellpose` installed"):
        st._get_cellpose_model()


# ---------------------------------------------------------------------------
# keeping every candidate edge instead of one per direction
# ---------------------------------------------------------------------------

def _pair_row(a, b, dx, dy, score):
    return {"pathA": a, "pathB": b, "dx_px_full": str(dx),
            "dy_px_full": str(dy), "theta_deg": "0.0", "scale": "1.0",
            "score": str(score)}


def test_uncapping_the_direction_keeps_every_candidate_edge(tmp_path):
    """``cap_one_per_dir=False`` must offer the spanning tree the alternatives.

    The cap keeps only the best-scoring edge per tile per direction. When that
    winner is a false match -- a repeated background pattern outscoring the
    true neighbour -- the tree is handed the wrong edge and nothing else, and
    the tile lands in the wrong place. Turning the cap off is the remedy, so
    the runner-up edges have to survive to be chosen.
    """
    st = _st(tmp_path)
    rows = [
        _pair_row("t1.tif", "t2.tif", 150, 0, 0.9),
        # A second, lower-scoring transform between the same tiles, in the
        # same direction: the impostor the cap would keep, or the true
        # neighbour it would throw away.
        _pair_row("t1.tif", "t2.tif", 152, 0, 0.4),
        _pair_row("t2.tif", "t3.tif", 150, 0, 0.8),
    ]

    _, capped = st._compute_mosaic_transforms(rows, min_score=0.1,
                                              cap_one_per_dir=True)
    transforms, uncapped = st._compute_mosaic_transforms(
        rows, min_score=0.1, cap_one_per_dir=False)

    assert {p for edge in uncapped for p in edge[:2]} == {"t1.tif", "t2.tif",
                                                          "t3.tif"}
    assert set(transforms) == {"t1.tif", "t2.tif", "t3.tif"}
    # The spanning tree still takes one edge per join; what changes is how
    # many it had to choose from.
    assert len(uncapped) == len(capped) == 2


def test_the_uncapped_run_offers_the_tree_more_edges_than_the_capped_one(
        tmp_path, capsys):
    """The parameter has to change what the spanning tree chooses among.

    Stated as a count because that is the whole mechanism: the cap collapses
    every edge leaving a tile in one direction down to the best-scoring one,
    so an impostor that outscores the true neighbour leaves no alternative
    behind. The run reports the candidate count it gated down to, which is
    where the difference is visible.
    """
    st = _st(tmp_path)
    st.verbose = True
    rows = [
        _pair_row("t1.tif", "t2.tif", 150, 0, 0.9),
        _pair_row("t1.tif", "t2.tif", 152, 0, 0.4),
        _pair_row("t2.tif", "t3.tif", 150, 0, 0.8),
    ]

    st._compute_mosaic_transforms(rows, min_score=0.1, cap_one_per_dir=True)
    capped_line = capsys.readouterr().out
    st._compute_mosaic_transforms(rows, min_score=0.1, cap_one_per_dir=False)
    uncapped_line = capsys.readouterr().out

    def _count(text):
        marker = "candidate edges after gating: "
        return int(text.split(marker)[1].split(";")[0])

    assert _count(uncapped_line) > _count(capped_line)
    assert _count(uncapped_line) == len(rows) * 2


# ---------------------------------------------------------------------------
# the mosaic builder's own channel count
# ---------------------------------------------------------------------------

_MANIFEST_COLUMNS = ["path", "H", "W", "M00", "M01", "M02", "M10", "M11",
                     "M12", "canvas_x", "canvas_y"]


def _manifest(tmp_path, tiles, offsets):
    """A minimal mosaic manifest placing each tile at a known offset."""
    target = str(tmp_path / "manifest.csv")
    with open(target, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        for path, (dx, dy) in zip(tiles, offsets):
            array = tifffile.imread(path)
            height, width = array.shape[-2:]
            writer.writerow({
                "path": path, "H": height, "W": width,
                "M00": 1.0, "M01": 0.0, "M02": dx,
                "M10": 0.0, "M11": 1.0, "M12": dy,
                "canvas_x": dx, "canvas_y": dy,
            })
    return target


def test_a_mosaic_of_axesless_multichannel_tiles_keeps_every_channel(
        tmp_path, axeless_reader):
    """With no axes metadata the builder counts channels from the shape.

    ``channel_indices=None`` means "mosaic everything this tile has". Getting
    that count wrong writes a mosaic with fewer planes than the tiles held,
    and nothing downstream can tell that a stain is simply not there.
    """
    planes = [[_plane(seed=20 + i * 2), _plane(seed=21 + i * 2)]
              for i in range(2)]
    tiles = [_write_stack(str(tmp_path / f"tile{i}.tif"), planes[i])
             for i in range(2)]
    manifest = _manifest(tmp_path, tiles, [(0, 0), (32, 0)])
    out = str(tmp_path / "mosaic.tif")

    written = _st(tmp_path).build_multichannel_mosaic_from_manifest(
        manifest, out, tmp_dir=str(tmp_path / "tmp"))

    assert written == out
    mosaic = tifffile.imread(out)
    assert mosaic.shape[0] == 2
    assert not np.array_equal(mosaic[0], mosaic[1])


def test_a_mosaic_of_axesless_single_plane_tiles_is_one_channel(
        tmp_path, axeless_reader):
    """A 2-D tile has one channel however little metadata came with it.

    The same count decides how big the workspace is; guessing more than one
    for a flat tile allocates empty planes and writes them into the output as
    if they were data.
    """
    tiles = [str(tmp_path / f"flat{i}.tif") for i in range(2)]
    for index, path in enumerate(tiles):
        tifffile.imwrite(path, _plane(seed=40 + index))
    manifest = _manifest(tmp_path, tiles, [(0, 0), (32, 0)])
    out = str(tmp_path / "flat_mosaic.tif")

    _st(tmp_path).build_multichannel_mosaic_from_manifest(
        manifest, out, tmp_dir=str(tmp_path / "tmp"))

    mosaic = tifffile.imread(out)
    assert mosaic.ndim == 2 or mosaic.shape[0] == 1
