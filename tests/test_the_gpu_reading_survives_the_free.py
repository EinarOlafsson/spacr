"""The GPU number is a HIGH-WATER MARK, not whatever is live at the boundary.

Stages are recorded at their boundaries and a fit frees its tensors before
the boundary is reached. Reading `torch.cuda.memory_allocated()` there gives
~0 no matter how large the fit was: measured on the maintainer's screen, a
mixed fit on the GPU backend recorded "PEAK GPU 0.0 B" -- the one number
instruction 160 exists to obtain, and the one reading that cannot be true.
"""
import pytest

from spacr.fit_resources import gpu_allocated, readable

torch = pytest.importorskip("torch")

# `gpu` AS WELL AS the CUDA check: these allocate 64 MB on the card, and the
# card is shared. `tests/conftest.py::pytest_runtest_setup` skips a
# gpu-marked test when another session has filled it, which is the ordinary
# state of this machine -- without the marker they fail on an allocation
# that has nothing to do with what they measure.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="the reading is a CUDA reading"),
]


def test_the_peak_is_still_reported_after_the_tensor_is_freed():
    before = gpu_allocated()
    block = torch.zeros(64 * 1024 * 1024 // 4, device="cuda")   # 64 MB
    held = gpu_allocated()
    del block
    torch.cuda.empty_cache()
    after = gpu_allocated()

    assert held >= (before or 0) + 60 * 1024 * 1024
    assert after >= held, (
        "the reading dropped when the memory was freed, so every stage "
        "boundary reports ~0 and the GPU column is useless")


def test_no_torch_import_is_forced(monkeypatch):
    # Importing torch to take a measurement would make the measurement the
    # most expensive thing in the stage.
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    assert gpu_allocated() is None


def test_unmeasured_is_words_not_a_number():
    assert readable(None) == "not measured"
    assert readable(0) == "0.0 B"
