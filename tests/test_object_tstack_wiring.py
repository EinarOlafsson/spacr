"""The 4D (Beta) `t_stack` setting, wired into ``spacr.object``.

Before this file existed, ``t_stack`` was declared, typed, tooltipped,
categorised and rendered a widget in both GUIs, and **nothing read it**:
``zstack.plan_4d_from_settings``, ``zstack.segment_4d`` and ``zstack.track_4d``
were called by nothing outside ``zstack.py`` and its own tests. Turning the
setting on changed nothing at all, so a user who believed they had enabled 4-D
got 2-D results with no indication. These tests pin the fix.

Everything here is CPU-only, offline and model-free: ``cellpose.models`` is
monkeypatched with the same deterministic fake ``test_zstack.py`` and
``test_cov_object_masks_sam.py`` use, so no network is loaded and nothing is
downloaded.

The load-bearing tests, in the order they matter:

1. ``test_2d_is_untouched_*`` and ``test_the_3d_path_is_untouched_*`` -- the
   acceptance criterion. A user who does not opt into 4-D must not see any
   change at all, so the masks on disk, the rows in the database and the kwargs
   handed to Cellpose are compared byte for byte between a run whose settings
   have never heard of t and a run carrying every 4D key with ``t_stack`` off.
2. ``test_t_stack_on_without_a_z_axis_names_the_ingest_as_the_cause`` -- opting
   in when the ingest has already projected z away stops the run naming
   ``io._rename_and_organize_image_files``, rather than segmenting frame by
   frame and reporting a 4-D result.
3. ``test_t_stack_on_without_an_axis_order_refuses_rather_than_guessing`` --
   the ``(T,Z,Y,X)`` / ``(Z,T,Y,X)`` ambiguity is surfaced naming
   ``t_axis_order``, never resolved by a guess.
4. ``test_the_settings_object_reads_are_exactly_the_ones_settings_declares`` --
   a renamed key cannot silently stop being read again.
"""
from __future__ import annotations

import sqlite3
import types
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import spacr.object as O
import spacr.settings as S
import spacr.zstack as Z

from tests.cellpose_api_contract import (
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call


# The 4D (Beta) block as ``spacr.settings`` declares it. Kept as a literal so
# that renaming a key in settings.py without renaming it in zstack.py's
# settings bridge fails here instead of silently going unread.
FOUR_D_KEYS = {
    "t_stack", "t_axis_order", "t_axis", "frame_interval_s", "t_track_backend",
    "t_link_threshold", "t_max_displacement_px", "t_max_displacement_um",
    "t_project_for_tracking",
}

# The z keys a 4-D plan shares with a 3-D one. `plan_4d_from_settings` reads
# these rather than owning a second spelling of the same physics.
SHARED_Z_KEYS = {
    "z_axis", "z_segmentation_mode", "z_projection", "anisotropy",
    "voxel_size_z_um", "voxel_size_xy_um", "stitch_threshold",
}


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


@pytest.fixture
def fake_model(monkeypatch):
    """A deterministic stand-in for ``cellpose.models.CellposeModel``.

    Handles both call shapes spaCR uses: a list of 2-D images (the ordinary
    path, the per-plane stitch path and the projected path) and a single
    ``(Z, Y, X, C)`` volume (the volumetric path). No real model, no download.
    """
    holder = {"model": None}

    class _M:
        """``CellposeModel`` double declaring the installed 4.0.7 signatures.

        No ``**kwargs``: ``generate_cellpose_masks_sam`` is a real call site and
        the 2-D/3-D/4-D dispatch under test builds its eval kwargs by hand, so
        a stray or renamed argument has to raise ``TypeError`` here.

        ``eval_kwargs`` holds every bound parameter; ``eval_configured`` holds
        only those spaCR set away from cellpose's default, which is what "the
        4D settings must not leak into the 2-D call" actually means.
        """

        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None,
                     use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.device = device
            self.init_kwargs = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                         model_type)
            self.eval_kwargs = []
            self.eval_configured = []
            self.eval_shapes = []
            holder["model"] = self

        @staticmethod
        def _label(image):
            out = np.zeros(image.shape[:2], dtype=np.uint16)
            out[2:8, 2:8] = 1
            out[12:18, 12:18] = 2
            return out

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            # Every one of these call sites names channel_axis, and the 3-D one
            # also names z_axis/do_3D -- all of which convert_image reads, so
            # the whole axis combination goes through the real validator.
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            bound = locals()
            self.eval_kwargs.append(eval_arguments(bound))
            self.eval_configured.append(configured_eval_arguments(bound))
            if isinstance(x, list):
                self.eval_shapes.append([np.asarray(i).shape for i in x])
                masks = [self._label(np.asarray(i)) for i in x]
                flows = [np.zeros(m.shape, np.float32) for m in masks]
                # THREE values -- cellpose 4's (masks, flows, styles). This
                # returned four, which is the cellpose 3 shape.
                return masks, flows, None
            volume = np.asarray(x)
            self.eval_shapes.append(volume.shape)
            labels = np.stack([self._label(volume[z])
                               for z in range(volume.shape[0])])
            return labels, [np.zeros(labels.shape, np.float32)], None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_M))
    return holder


def _write_npz(src: Path, shape, name="batch1.npz", seed=0):
    """Write one pre-batched npz.

    ``shape`` is ``(N, Y, X, C)`` for the 2-D path, ``(N, Z, Y, X, C)`` for the
    3-D one and ``(T, Z, Y, X, C)`` for the 4-D one -- the last two are the
    same array layout, and ``t_stack`` is what declares the leading axis to be
    time rather than a list of independent fields.
    """
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4000, size=shape).astype(np.uint16)
    filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(shape[0])])
    np.savez(src / name, data=data, filenames=filenames)
    return data


def _base_settings(src, **over):
    settings = {
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
        "seg_qc": "off",
    }
    settings.update(over)
    return settings


def _artifacts(src, object_type="cell"):
    """Everything a run leaves behind, for a byte-for-byte comparison."""
    folder = Path(src) / f"{object_type}_mask_stack"
    masks = {p.name: p.read_bytes() for p in sorted(folder.iterdir())}

    db = Path(src).parent / "measurements" / "measurements.db"
    con = sqlite3.connect(str(db))
    try:
        rows = sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts"))
    finally:
        con.close()
    return masks, rows


class _RecordingSettings(dict):
    """A settings dict that remembers which keys were actually looked up."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


# ===========================================================================
# 1. The acceptance criterion: off must mean nothing happened
# ===========================================================================

def test_2d_is_untouched_when_the_4d_settings_are_absent_or_present_but_off(
        tmp_path, fake_model):
    """A user who does not opt into 4-D must not see any change at all.

    Two runs on identical input: one whose settings have never heard of t, one
    carrying every 4D key explicitly set but with ``t_stack`` off (and the rest
    at deliberately provocative values -- a backend spaCR cannot drive on
    volumes, both displacement gates, an axis order). The masks on disk, the
    rows in the database and the kwargs handed to Cellpose must all match
    exactly.
    """
    src_a = tmp_path / "a" / "stack"
    src_b = tmp_path / "b" / "stack"
    _write_npz(src_a, (3, 32, 32, 2), seed=7)
    _write_npz(src_b, (3, 32, 32, 2), seed=7)

    O.generate_cellpose_masks_sam(str(src_a), _base_settings(src_a), "cell")
    kwargs_a = [dict(k) for k in fake_model["model"].eval_kwargs]

    O.generate_cellpose_masks_sam(str(src_b), _base_settings(
        src_b,
        t_stack=False,                  # the switch, off
        t_axis_order="ZTYX",
        t_axis=1,
        frame_interval_s=30.0,
        t_track_backend="btrack",
        t_link_threshold=0.9,
        t_max_displacement_px=5.0,
        t_max_displacement_um=2.0,
        t_project_for_tracking=True,
    ), "cell")
    kwargs_b = [dict(k) for k in fake_model["model"].eval_kwargs]

    masks_a, rows_a = _artifacts(src_a)
    masks_b, rows_b = _artifacts(src_b)

    assert masks_a.keys() == masks_b.keys()
    for name in masks_a:
        assert masks_a[name] == masks_b[name], (
            f"{name} differs: turning the 4D settings on-but-off changed the "
            f"2-D masks"
        )
    assert rows_a == rows_b
    assert kwargs_a == kwargs_b, (
        "Cellpose was called differently; the 4D settings must not leak into "
        "the 2-D eval call"
    )
    # "Not passed" now means "left at cellpose's own default", because the
    # double declares the whole signature rather than collecting **kwargs.
    for kwargs in fake_model["model"].eval_configured:
        assert "do_3D" not in kwargs
        assert "anisotropy" not in kwargs
        assert "z_axis" not in kwargs


def test_the_3d_path_is_untouched_when_t_stack_is_off(tmp_path, fake_model):
    """#28's path must be bit-identical with the 4D keys present but off.

    ``z_stack`` on, ``t_stack`` off, once with no 4D keys at all and once with
    every one of them set to something provocative.
    """
    src_a = tmp_path / "a" / "stack"
    src_b = tmp_path / "b" / "stack"
    _write_npz(src_a, (2, 4, 32, 32, 2), seed=11)
    _write_npz(src_b, (2, 4, 32, 32, 2), seed=11)

    z_only = dict(z_stack=True, z_segmentation_mode="volumetric",
                  voxel_size_z_um=5.0, voxel_size_xy_um=1.0)

    O.generate_cellpose_masks_sam(str(src_a), _base_settings(src_a, **z_only),
                                  "cell")
    kwargs_a = [dict(k) for k in fake_model["model"].eval_kwargs]

    O.generate_cellpose_masks_sam(str(src_b), _base_settings(
        src_b,
        t_stack=False,
        t_axis_order="ZTYX",
        frame_interval_s=30.0,
        t_track_backend="btrack",
        t_link_threshold=0.9,
        t_project_for_tracking=True,
        **z_only,
    ), "cell")
    kwargs_b = [dict(k) for k in fake_model["model"].eval_kwargs]

    masks_a, rows_a = _artifacts(src_a)
    masks_b, rows_b = _artifacts(src_b)

    assert masks_a.keys() == masks_b.keys()
    for name in masks_a:
        assert masks_a[name] == masks_b[name], (
            f"{name} differs: the 4D keys leaked into the 3-D path"
        )
    assert rows_a == rows_b
    assert kwargs_a == kwargs_b


def test_the_plan_is_none_when_the_setting_is_off_so_no_4d_code_runs():
    """The contract every call site branches on."""
    assert O._t_stack_plan({}) is None
    assert O._t_stack_plan({"t_stack": False, "t_axis_order": "TZYX"}) is None
    assert O._t_stack_plan(
        S.set_default_settings_preprocess_generate_masks({})) is None


def test_the_plans_are_not_reconciled_at_all_when_t_is_off(capsys):
    """With ``t_stack`` off, ``_reconcile_z_and_t_plans`` is the identity."""
    z_plan = Z.ZStackSpec(mode="stitch")
    assert O._reconcile_z_and_t_plans(z_plan, None) is z_plan
    assert O._reconcile_z_and_t_plans(None, None) is None
    assert O._reconcile_z_and_t_plans(z_plan, None, timelapse=True) is z_plan
    assert capsys.readouterr().out == "", "an off setting must say nothing"


# ===========================================================================
# 2. Opting in without the data: the run stops and says why
# ===========================================================================

def test_t_stack_on_without_a_z_axis_names_the_setting_and_the_cause(
        tmp_path, fake_model):
    """Silently segmenting frame by frame and calling it 4-D is the one
    outcome worse than having no 4-D at all."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 32, 32, 2))

    settings = _base_settings(src, t_stack=True, t_axis_order="TZYX")

    with pytest.raises(Z.TAxisNotPresentError) as excinfo:
        O.generate_cellpose_masks_sam(str(src), settings, "cell")

    message = str(excinfo.value)
    assert "t_stack" in message, "the message must name the setting"
    assert "no z axis" in message
    assert "io._rename_and_organize_image_files" in message, (
        "the message must name where the axis was lost, not just that it is "
        "missing"
    )
    assert "t_stack off" in message, "and how to proceed"
    assert "timelapse" in message, (
        "a flat 2-D time series has a path that works today and it must be "
        "pointed at"
    )
    assert "segment_4d" in message, "and so must the Python API that does work"


def test_the_flat_array_error_is_raised_before_any_field_is_segmented(
        tmp_path, fake_model):
    """It cannot become true later in the run, so it is answered first."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 32, 32, 2))

    with pytest.raises(Z.TAxisNotPresentError):
        O.generate_cellpose_masks_sam(
            str(src), _base_settings(src, t_stack=True, t_axis_order="TZYX"),
            "cell")

    assert fake_model["model"].eval_kwargs == [], (
        "Cellpose was driven before the run discovered it could not proceed"
    )
    assert not (src / "cell_mask_stack").exists() or not list(
        (src / "cell_mask_stack").iterdir()), "and it wrote no masks"


def test_the_guard_accepts_a_real_four_dimensional_batch():
    """``_require_t_axis`` is a shape check and nothing more."""
    plan = O._t_stack_plan({"t_stack": True, "t_axis_order": "TZYX"})
    assert O._require_t_axis(np.zeros((3, 4, 32, 32, 2)), plan, "b.npz") is None
    with pytest.raises(Z.TAxisNotPresentError):
        O._require_t_axis(np.zeros((3, 32, 32, 2)), plan, "b.npz")


# ===========================================================================
# 3. The axis order is refused, never guessed
# ===========================================================================

def test_t_stack_on_without_an_axis_order_refuses_rather_than_guessing(
        tmp_path, fake_model):
    """``(T,Z,Y,X)`` and ``(Z,T,Y,X)`` are both real and the shape cannot say
    which you have; reading one as the other reports z as motion."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 4, 32, 32, 2))

    settings = _base_settings(src, t_stack=True)

    with pytest.raises(Z.AmbiguousAxisOrderError) as excinfo:
        O.generate_cellpose_masks_sam(str(src), settings, "cell")

    message = str(excinfo.value)
    assert "t_axis_order" in message, (
        "the message must name the setting that settles it"
    )
    assert "TZYX" in message and "ZTYX" in message, "and both readings"
    assert "not guess" in message
    assert fake_model["model"] is None, (
        "the model must not even be loaded before an unanswerable question is "
        "answered"
    )


def test_the_ambiguity_is_refused_at_plan_time_by_the_helper_too():
    """The same refusal, reachable without a whole pipeline run."""
    with pytest.raises(Z.AmbiguousAxisOrderError):
        O._t_stack_plan({"t_stack": True})

    # A leading z_axis settles it; a non-leading one cannot and says so.
    assert O._t_stack_plan({"t_stack": True, "z_axis": 1}).t_axis == 0
    with pytest.raises(Z.TStackError) as excinfo:
        O._t_stack_plan({"t_stack": True, "z_axis": 2})
    assert "t_axis_order" in str(excinfo.value)


def test_an_order_that_contradicts_an_explicit_axis_is_refused():
    """Two settings that disagree are two chances to be wrong, not one."""
    with pytest.raises(Z.TStackError) as excinfo:
        O._t_stack_plan({"t_stack": True, "t_axis_order": "TZYX", "t_axis": 1})
    assert "not pick one" in str(excinfo.value)


# ===========================================================================
# 4. Opted in with real 4-D data: the path reaches zstack.segment_4d
# ===========================================================================

def test_an_explicit_order_reaches_segment_4d_with_the_acquisition_and_spec(
        tmp_path, fake_model, monkeypatch):
    """The wiring itself, asserted against a mock rather than a model."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 4, 32, 32, 2))

    calls = []
    real_segment_4d = Z.segment_4d

    def _spy(array, spec, segment_fn, verbose=False):
        calls.append({"shape": np.asarray(array).shape, "spec": spec,
                      "segment_fn": segment_fn})
        return real_segment_4d(array, spec,
                               lambda plane, **kw: np.zeros(
                                   np.asarray(plane).shape[:2], np.uint16))

    monkeypatch.setattr(Z, "segment_4d", _spy)

    settings = _base_settings(src, t_stack=True, t_axis_order="TZYX")
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(calls) == 1, "one batch is one acquisition, not one per field"
    call = calls[0]
    assert call["shape"] == (3, 4, 32, 32, 2), (
        "segment_4d must get the whole (T, Z, Y, X, C) acquisition"
    )
    spec = call["spec"]
    assert isinstance(spec, Z.TStackSpec)
    assert (spec.t_axis, spec.z_axis) == (0, 1), "as t_axis_order='TZYX' says"
    assert spec.z_mode == "project", "the shared z default"


def test_the_zt_order_is_carried_through_to_the_spec(tmp_path, fake_model,
                                                     monkeypatch):
    """'ZTYX' must reach segment_4d as ``t_axis=1``, not be normalised away."""
    src = tmp_path / "stack"
    _write_npz(src, (4, 3, 32, 32, 2))

    seen = {}
    real_segment_4d = Z.segment_4d

    def _spy(array, spec, segment_fn, verbose=False):
        seen["spec"] = spec
        return real_segment_4d(array, spec,
                               lambda plane, **kw: np.zeros(
                                   np.asarray(plane).shape[:2], np.uint16))

    monkeypatch.setattr(Z, "segment_4d", _spy)

    O.generate_cellpose_masks_sam(
        str(src), _base_settings(src, t_stack=True, t_axis_order="ZTYX"),
        "cell")

    assert (seen["spec"].t_axis, seen["spec"].z_axis) == (1, 0)
    assert seen["spec"].axis_order == Z.AXIS_ORDER_ZTYX


def test_a_volumetric_4d_run_drives_cellpose_once_per_timepoint(tmp_path,
                                                                fake_model):
    """End to end through the real segment_4d, with a fake segmenter."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 5, 32, 32, 2))       # (T, Z, Y, X, C)

    settings = _base_settings(
        src, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="volumetric",
        voxel_size_z_um=5.0, voxel_size_xy_um=1.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    assert len(model.eval_kwargs) == 3, "one call per timepoint"
    for kwargs in model.eval_kwargs:
        assert kwargs["do_3D"] is True
        assert kwargs["anisotropy"] == 5.0, "derived from the voxel size"
        assert kwargs["z_axis"] == 0
        assert kwargs["channel_axis"] == -1
    assert model.eval_shapes == [(5, 32, 32, 2)] * 3, (
        "each call saw one whole (Z, Y, X, C) timepoint"
    )

    written = sorted((src / "cell_mask_stack").iterdir())
    assert len(written) == 3, "one mask per timepoint"
    for path in written:
        assert np.load(path).shape == (5, 32, 32), "and it kept its z axis"


def test_a_projected_4d_run_gives_one_2d_mask_per_timepoint(tmp_path,
                                                            fake_model):
    """'project' is the only mode spacr.measure can consume, in 4-D as in 3-D."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 4, 32, 32, 2))

    settings = _base_settings(
        src, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="project", z_projection="max",
        cell_min_object_area=1,          # force merge/split/filter to run
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    written = sorted((src / "cell_mask_stack").iterdir())
    assert len(written) == 3
    for path in written:
        assert np.load(path).shape == (32, 32)


def test_project_mode_filters_against_the_projection_it_segmented(
        tmp_path, fake_model, capsys):
    """merge/split/filter scores masks against intensities, so it must see the
    plane the masks were drawn on, not the volume it came from."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 4, 32, 32, 2))

    settings = _base_settings(
        src, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="project", cell_min_object_area=1,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    # Reaching here at all is the assertion: handing merge_split_filter_masks
    # the raw (T, Z, Y, X, C) acquisition raises "Unsupported intensity_images
    # ndim: 5".
    assert "merge_split_filter_masks(cell): skipped" not in out
    assert "perimeter_merge" in out, "the filter step really ran"


def test_the_volumetric_4d_modes_skip_the_2d_merge_split_filter_step(
        tmp_path, fake_model, capsys):
    """Applying a 2-D area filter per plane would tear the labels apart."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 4, 32, 32, 2))

    settings = _base_settings(
        src, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="volumetric", anisotropy=2.0,
        cell_min_object_area=1,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "merge_split_filter_masks(cell): skipped" in out
    assert "2-D only" in out
    assert "z_segmentation_mode='volumetric'" in out


def test_plotting_is_skipped_rather_than_crashing_in_4d(tmp_path, fake_model,
                                                        capsys):
    """The 4-D path calls eval per timepoint and collects no flow field."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 4, 32, 32, 2))

    settings = _base_settings(
        src, plot=True, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="volumetric", anisotropy=2.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "plot skipped" in out
    assert "volumetric" in out


def test_verbose_reports_what_the_4d_run_actually_did(tmp_path, fake_model,
                                                      capsys):
    """A number without the mode that produced it cannot be compared."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 4, 32, 32, 2))

    settings = _base_settings(
        src, verbose=True, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="stitch",
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "[4D]" in out
    assert "timepoint(s)" in out
    assert "track_4d has linked them" in out, (
        "the user must be told that segmentation is not tracking"
    )


def test_the_4d_adapter_is_the_very_same_cellpose_adapter_the_3d_path_uses():
    """A 4-D run and a 3-D run must not be able to drift apart in how they
    drive the model, so there is exactly one adapter."""
    import inspect

    source = inspect.getsource(O._segment_timepoints_with_t)
    assert "_cellpose_z_segment_fn" in source
    assert "model.eval" not in source, "no second Cellpose call site"


# ===========================================================================
# 5. t on / z off, and the reverse
# ===========================================================================

def test_t_stack_on_with_z_stack_off_still_drives_the_4d_path(tmp_path,
                                                              fake_model):
    """``t_stack`` is self-sufficient: ``plan_4d_from_settings`` reads the z
    keys itself and never consults ``z_stack``, so 4-D does not require the
    3-D switch as well."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 4, 32, 32, 2))

    settings = _base_settings(
        src, t_stack=True, z_stack=False, t_axis_order="TZYX",
        z_segmentation_mode="volumetric", anisotropy=3.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    assert len(model.eval_kwargs) == 3, "one call per timepoint"
    assert all(k["do_3D"] is True for k in model.eval_kwargs)
    assert all(k["anisotropy"] == 3.0 for k in model.eval_kwargs)


def test_a_flat_time_series_is_expressible_as_t_axis_order_TYX():
    """A 2-D movie is declarable, and only by saying so explicitly.

    Every other spelling of ``t_axis_order`` claims a z axis: ``TZYX``/``ZTYX``
    name both, and a bare ``t_axis`` derives ``z_axis = 1 - t_axis``. ``TYX``
    is the one spelling that means "there is no z", so a flat acquisition is
    never the result of an ambiguity being resolved in its favour — the user
    has to state it.
    """
    assert O._t_stack_plan({"t_stack": True, "t_axis_order": "TZYX"}).z_axis == 1
    assert O._t_stack_plan({"t_stack": True, "t_axis": 0}).z_axis == 1
    assert O._t_stack_plan({"t_stack": True, "t_axis": 1}).z_axis == 0

    flat = O._t_stack_plan({"t_stack": True, "t_axis_order": "TYX"})
    assert flat.z_axis is None
    assert flat.t_axis == 0

    # TYX plus a z axis is a contradiction, and is refused rather than one of
    # the two being quietly dropped.
    with pytest.raises(Z.TStackError):
        O._t_stack_plan({"t_stack": True, "t_axis_order": "TYX", "z_axis": 1})


def test_object_handles_a_flat_spec_end_to_end():
    """``object.py``'s half of the flat case, now reachable from settings too."""
    flat = Z.TStackSpec(t_axis=0, z_axis=None)

    # Four axes is enough for (T, Y, X, C) -- no z is missing, because none
    # was claimed.
    assert O._require_t_axis(np.zeros((3, 32, 32, 2)), flat, "b.npz") is None
    with pytest.raises(Z.TAxisNotPresentError):
        O._require_t_axis(np.zeros((32, 32, 2)), flat, "b.npz")

    class _Model:
        """Installed cellpose 4.0.7 ``eval`` signature, three-value return."""

        def __init__(self):
            self.shapes = []

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            images = x if isinstance(x, list) else [x]
            self.shapes.extend(np.asarray(i).shape for i in images)
            masks = [np.zeros(np.asarray(i).shape[:2], np.uint16)
                     for i in images]
            return masks, [np.zeros(m.shape, np.float32) for m in masks], None

    model = _Model()
    # The eval kwargs generate_cellpose_masks_sam actually builds for this
    # path (object.py's z_eval_kwargs), not an empty dict: channel_axis=-1 is
    # part of how spaCR drives Cellpose here, and the double now holds it to
    # the same convert_image contract the real eval applies.
    masks, result, intensity = O._segment_timepoints_with_t(
        np.zeros((3, 16, 16, 2), np.float32), model, flat,
        {"batch_size": 1, "normalize": False, "channel_axis": -1})

    assert len(masks) == 3 and all(m.shape == (16, 16) for m in masks)
    assert model.shapes == [(16, 16, 2)] * 3, "one plain 2-D call per frame"
    assert intensity is None, (
        "there is no z to collapse, so there is no projected copy either and "
        "the caller scores against the batch it already has"
    )
    assert result.z_mode == Z.MODE_SINGLE_PLANE

    # And the legacy 2-D trackers are perfectly happy with 2-D masks.
    assert O._reconcile_z_and_t_plans(None, flat, timelapse=True) is None


def test_z_stack_on_with_t_stack_off_is_the_unchanged_3d_path(tmp_path,
                                                              fake_model):
    """#28's path, untouched: the leading axis is fields, not timepoints, and
    nothing 4-D executes."""
    src = tmp_path / "stack"
    _write_npz(src, (2, 4, 32, 32, 2))

    settings = _base_settings(
        src, z_stack=True, t_stack=False, z_segmentation_mode="volumetric",
        anisotropy=3.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(fake_model["model"].eval_kwargs) == 2, "one call per field"
    assert O._t_stack_plan(settings) is None
    assert O._z_stack_plan(settings) is not None


def test_both_switches_on_lets_the_4d_plan_supersede_the_3d_one(tmp_path,
                                                                fake_model,
                                                                capsys):
    """segment_4d already runs segment_3d per timepoint with these very same z
    settings, so leaving both live would segment every field twice."""
    src = tmp_path / "stack"
    _write_npz(src, (3, 4, 32, 32, 2))

    settings = _base_settings(
        src, z_stack=True, t_stack=True, t_axis_order="TZYX",
        z_segmentation_mode="volumetric", anisotropy=2.0,
    )
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "both on" in out
    assert "supersedes" in out
    assert "twice" in out
    # 3 timepoints, not 3 timepoints plus 3 fields.
    assert len(fake_model["model"].eval_kwargs) == 3


def test_the_supersede_rule_is_stated_by_the_helper_directly(capsys):
    """The z plan is dropped, and the drop is announced."""
    z_plan = Z.ZStackSpec(mode="stitch")
    t_plan = Z.TStackSpec(t_axis=0, z_axis=1, z_mode="stitch")

    assert O._reconcile_z_and_t_plans(z_plan, t_plan) is None
    assert "supersedes" in capsys.readouterr().out

    # And with no z plan there is nothing to announce.
    assert O._reconcile_z_and_t_plans(None, t_plan) is None
    assert capsys.readouterr().out == ""


def test_legacy_timelapse_tracking_on_4d_volumes_is_refused_not_flattened():
    """Every timelapse adapter requires (T, Y, X); projecting z away to make
    them accept a volume would produce a track table nothing can audit."""
    t_plan = Z.TStackSpec(t_axis=0, z_axis=1, z_mode="volumetric",
                          anisotropy=2.0)

    with pytest.raises(Z.TrackerIsTwoDError) as excinfo:
        O._reconcile_z_and_t_plans(None, t_plan, timelapse=True)

    message = str(excinfo.value)
    assert "timelapse" in message
    assert "_btrack_track_cells" in message, "name the adapters that refuse"
    assert "z_segmentation_mode='project'" in message, "and the way out"
    assert "track_4d" in message, "and the linker that does do 3-D"

    # 'project' keeps the masks 2-D, so the legacy trackers are fine with it.
    flat = Z.TStackSpec(t_axis=0, z_axis=1, z_mode="project")
    assert O._reconcile_z_and_t_plans(None, flat, timelapse=True) is None


# ===========================================================================
# 6. The generators that cannot honour it say so
# ===========================================================================

def test_a_generator_without_a_4d_path_refuses_rather_than_ignoring():
    """A 4-D setting must never be accepted for a quietly 2-D result."""
    for where in ("object.generate_cellpose_masks",
                  "object.generate_organelle_masks_sam"):
        with pytest.raises(Z.TStackError) as excinfo:
            O._refuse_t_stack({"t_stack": True}, where)
        message = str(excinfo.value)
        assert where in message
        assert "generate_cellpose_masks_sam" in message, "name what does work"
        assert "quietly return a 2-D result" in message

    # Off, and absent, are both silent no-ops.
    assert O._refuse_t_stack({"t_stack": False}, "x") is None
    assert O._refuse_t_stack({}, "x") is None


def test_the_organelle_generator_refuses_before_it_touches_anything(tmp_path):
    """The guard sits at the top, next to the defaults."""
    src = tmp_path / "stack"
    src.mkdir(parents=True)

    with pytest.raises(Z.TStackError):
        O.generate_organelle_masks_sam(
            str(src), {"src": str(src), "t_stack": True, "organelle_channel": 0},
            "organelle")

    assert not (src / "organelle_mask_stack").exists()


# ===========================================================================
# 7. The keys read are the keys declared
# ===========================================================================

def test_the_settings_object_reads_are_exactly_the_ones_settings_declares():
    """A renamed key must not be able to silently stop being read again.

    ``t_stack`` spent a whole release declared, typed, tooltipped and rendered
    while nothing read it. This asserts the other direction too: every key the
    4-D plan consults is a key ``settings.expected_types`` knows about, so a
    typo in the settings bridge is a failure here rather than a setting that
    quietly does nothing.
    """
    settings = _RecordingSettings(
        S.set_default_settings_preprocess_generate_masks({}))
    settings["t_stack"] = True
    settings["t_axis_order"] = "TZYX"
    settings.read.clear()

    plan = O._t_stack_plan(settings)
    assert plan is not None

    missing = FOUR_D_KEYS - settings.read
    assert not missing, (
        f"declared in settings.py and rendered in both GUIs but never read: "
        f"{sorted(missing)}"
    )

    shared_missing = SHARED_Z_KEYS - settings.read
    assert not shared_missing, (
        f"a 4-D run must read the same z keys a 3-D run does, but these were "
        f"not consulted: {sorted(shared_missing)}"
    )

    undeclared = {k for k in settings.read if k not in S.expected_types}
    assert not undeclared, (
        f"the 4-D plan reads keys settings.py does not declare: "
        f"{sorted(undeclared)}"
    )


def test_every_4d_key_is_declared_typed_and_categorised():
    """The four places a setting has to exist, checked together."""
    for key in FOUR_D_KEYS:
        assert key in S.expected_types, f"{key} has no declared type"
        assert key in S.categories["4D Settings (Beta)"], (
            f"{key} is in no GUI category")
        assert key in S.tooltips, f"{key} has no tooltip"

    defaults = S.set_default_settings_preprocess_generate_masks({})
    for key in FOUR_D_KEYS:
        assert key in defaults, f"{key} has no default"


def test_the_4d_tooltip_is_honest_about_where_the_run_stops():
    """A user must not read this and believe spaCR tracks volumes today."""
    tooltip = S.tooltips["t_stack"]
    assert "stops with an error" in tooltip
    assert "maximum-intensity" in tooltip or "collapses" in tooltip
