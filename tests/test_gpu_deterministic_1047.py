"""The opt-in deterministic GPU mode (item 1047, setting ``gpu_deterministic``).

The default builds and runs as before and touches no torch flag. Switched on,
the one shared build and evaluation switch the torch flags and the cuBLAS
workspace variable, and an operation torch refuses is reported by name of the
setting. The GPU test runs a small real cpsam_v2 segmentation twice and needs
the card and the pinned weights.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr import gpu_determinism, ops_engine, utils
from spacr.gpu_determinism import GpuDeterminismError
from spacr.ops_settings import ops_defaults
from spacr.settings import set_default_settings_preprocess_generate_masks


class _FakeModel:
    """A model that loads nothing, standing in for ``CellposeModel``."""

    def __init__(self, **kwargs):
        """Keep the keyword arguments of the build."""
        self.kwargs = kwargs


@pytest.fixture
def torch_flags(monkeypatch):
    """Record the torch determinism state and restore it afterwards."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    yield
    torch.use_deterministic_algorithms(before[0])
    torch.backends.cudnn.deterministic = before[1]
    torch.backends.cudnn.benchmark = before[2]


def _state():
    """The three torch flags and the workspace variable."""
    return (torch.are_deterministic_algorithms_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def test_default_is_off_and_nothing_is_set(torch_flags):
    """No setting, False, or the text false build a model and leave torch alone."""
    for settings in (None, {}, {"gpu_deterministic": False}, {"gpu_deterministic": "false"}):
        model = utils.build_cellpose_model("cpsam", settings=settings, gpu=False,
                                           factory=_FakeModel)
        assert isinstance(model, _FakeModel)
        assert _state() == (False, False, True, None)
    assert set_default_settings_preprocess_generate_masks({})["gpu_deterministic"] is False
    assert ops_defaults({})["gpu_deterministic"] is False


def test_on_sets_the_torch_flags_at_build(torch_flags):
    """A build with the mode on sets the three flags and the cuBLAS workspace."""
    utils.build_cellpose_model("cpsam", settings={"gpu_deterministic": True}, gpu=False,
                               factory=_FakeModel)
    assert _state() == (True, True, False, ":4096:8")


def test_ops_model_build_goes_through_the_same_switch(torch_flags):
    """The OPS model builder reaches the shared switch through the shared build."""
    from cellpose import models

    real = models.CellposeModel
    models.CellposeModel = _FakeModel
    try:
        ops_engine._cellpose_model({"gpu_deterministic": True, "ops_cellpose_model": "cpsam_v1"},
                                   False)
    except Exception:
        pass
    finally:
        models.CellposeModel = real
    assert torch.are_deterministic_algorithms_enabled()


def test_a_caller_workspace_value_is_kept(torch_flags, monkeypatch):
    """A cuBLAS workspace value the caller already set is not overwritten."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    assert gpu_determinism.apply_gpu_deterministic({"gpu_deterministic": True})
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_refusal_names_the_setting(torch_flags):
    """A model whose operation has no deterministic version stops with a clear error."""
    from spacr.object import _mask_eval

    class _Refusing:
        """A model whose evaluation hits an operation torch will not run deterministically."""

        def eval(self, *args, **kwargs):
            """Raise what torch raises for such an operation."""
            raise RuntimeError("scatter_add_cuda_kernel does not have a deterministic "
                               "implementation")

    with pytest.raises(GpuDeterminismError, match="gpu_deterministic"):
        _mask_eval(_Refusing(), np.zeros((8, 8)), site="t",
                   settings={"gpu_deterministic": True})
    with pytest.raises(RuntimeError) as plain:
        _mask_eval(_Refusing(), np.zeros((8, 8)), site="t", settings={})
    assert not isinstance(plain.value, GpuDeterminismError)


def test_provenance_records_the_mode():
    """The well provenance block carries the mode, off by default."""
    assert ops_engine._well_provenance({}, ("stitch",), {})["gpu_deterministic"] is False
    assert ops_engine._well_provenance({"gpu_deterministic": True}, ("stitch",),
                                       {})["gpu_deterministic"] is True


def _nuclei_image(seed=0):
    """A noisy 384 x 384 image of about forty bright discs."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:384, :384]
    image = rng.normal(100, 8, (384, 384))
    for _ in range(40):
        cy, cx = rng.integers(16, 368, 2)
        image += 900 * (((yy - cy) ** 2 + (xx - cx) ** 2) < rng.integers(36, 81))
    return image.astype(np.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_two_cpsam_v2_runs_give_byte_identical_masks(torch_flags):
    """Two runs of a small real cpsam_v2 segmentation agree byte for byte."""
    from spacr.object import _mask_eval
    from spacr.model_zoo import ModelUnreadable

    settings = {"gpu_deterministic": True, "ops_cellpose_model": "cpsam_v2"}
    try:
        model = ops_engine._cellpose_model(settings, True)
    except ModelUnreadable as error:
        pytest.skip(str(error))
    image = _nuclei_image()
    runs = []
    for _ in range(2):
        out = _mask_eval(model, image, site="t1047", settings=settings, batch_size=8,
                         diameter=10.0)
        runs.append(np.ascontiguousarray(out[0]))
    assert runs[0].max() > 5
    assert runs[0].tobytes() == runs[1].tobytes()
    assert _state()[:3] == (True, True, False)
    print("GPU:", torch.cuda.get_device_name(0), "objects:", int(runs[0].max()))
