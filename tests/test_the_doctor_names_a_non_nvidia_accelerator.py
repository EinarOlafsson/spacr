"""The doctor stops calling a working AMD or Apple GPU "no GPU".

`check_gpu` diagnoses CUDA, and its first branch reads:

    "No NVIDIA driver and a CPU-only torch: spaCR will run, but
     segmentation and training will be very slow."

A stock macOS torch has no CUDA version at all, so that is the verdict
every Mac got -- including the one this was fixed on, where the AMD Radeon
it does not mention segments 139x faster than the CPU it is warning about.
Wrong by two orders of magnitude, and pointing at `nvidia-smi` for a card
that is not the problem.

The CUDA paths must be untouched, which is most of what these assert:
every diagnostic below the new branch still has to run on an NVIDIA
machine, including the allocation probe that catches a driver mismatch
`torch.cuda.is_available()` reports as fine.
"""
from __future__ import annotations

import types

import pytest

from spacr import doctor
from spacr.doctor import FAIL, PASS, WARN


@pytest.fixture
def ctx():
    """A context with the allocation probe left on, as the CLI has it."""
    import inspect

    parameters = inspect.signature(doctor.Context).parameters
    return doctor.Context(**{name: value.default
                             for name, value in parameters.items()
                             if value.default is not inspect.Parameter.empty})


def _torch(cuda=False, cuda_build=None, hip=None, mps=False, name="GPU"):
    """A stand-in exposing only what the resolver and the doctor probe."""
    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda=cuda_build, hip=hip)
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        device_count=lambda: 1 if cuda else 0,
        get_device_name=lambda index=0: name,
        init=lambda: None,
    )
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps,
                                  is_built=lambda: mps))
    fake.zeros = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the CPU path must not allocate on a device"))
    return fake


def test_a_mac_driving_an_amd_card_through_metal_passes(ctx, monkeypatch):
    """The whole point: a working non-NVIDIA GPU reads as working."""
    monkeypatch.setattr(doctor, "_import_torch", lambda: _torch(mps=True))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    monkeypatch.setattr("spacr.accelerator._metal_gpu_name",
                        lambda: "AMD Radeon Pro 5300")

    row = doctor.check_gpu(ctx)

    assert row.status == PASS, "a usable Metal GPU is not a CUDA failure"
    assert "AMD Radeon Pro 5300" in row.message
    assert "very slow" not in row.message
    assert "nvidia-smi" not in (row.fix or ""), (
        "nvidia-smi is advice about hardware this machine does not have")


def test_the_metal_row_says_what_is_still_on_the_cpu(ctx, monkeypatch):
    """Detected is not the same as accelerated, per task.

    cuML ships for CUDA only, so the reductions stay on the CPU on this
    machine however well the card works. Saying "GPU found" and stopping
    there is what makes a user blame their hardware for the speed of a
    thing spaCR never sent to it.
    """
    monkeypatch.setattr(doctor, "_import_torch", lambda: _torch(mps=True))
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)
    # AND THE CAPABILITY TABLE, which `check_gpu` reads for this half of the
    # answer. It calls `accelerator.capabilities()`, and that re-resolves the
    # REAL machine rather than looking at the torch being diagnosed -- so on
    # a CUDA box the row said "Metal GPU" while the details underneath it
    # reported cuML accelerated. The test passed only where the developer
    # happened to have no CUDA.
    from spacr import accelerator

    monkeypatch.setattr(accelerator, "capabilities", lambda: (
        ("Segmentation (Cellpose)", True, "on the GPU"),
        ("UMAP / t-SNE / clustering", False, "cuML is CUDA-only"),
    ))

    details = " ".join(doctor.check_gpu(ctx).details or ())

    assert "still on the CPU" in details
    assert "float64" in details, (
        "float64 raises on Metal; a run that needs it silently moves to "
        "the CPU and the reader should hear that here")


def test_an_nvidia_machine_is_diagnosed_exactly_as_before(ctx, monkeypatch):
    """THE CONFIGURATION EVERY CURRENT USER HAS.

    The new branch must not intercept CUDA. If it did, the allocation
    probe below it would stop running and a driver/runtime mismatch would
    start reporting as a pass -- the exact failure that probe exists for.
    """
    allocated = []

    fake = _torch(cuda=True, cuda_build="12.1", name="NVIDIA RTX 4090")
    fake.zeros = lambda *a, **k: allocated.append(k.get("device")) or _Tensor()
    fake.cuda.synchronize = lambda: None
    monkeypatch.setattr(doctor, "_import_torch", lambda: fake)
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.173.02")

    row = doctor.check_gpu(ctx)

    assert row.status == PASS
    assert "RTX 4090" in row.message
    assert allocated == ["cuda"], (
        "the allocation probe was skipped; a driver/runtime mismatch would "
        "now pass silently")


class _Tensor:
    """Just enough tensor for the doctor's probe."""

    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 0.0


def test_a_cpu_only_machine_still_gets_the_cpu_warning(ctx, monkeypatch):
    """No accelerator of any vendor is still worth warning about."""
    monkeypatch.setattr(doctor, "_import_torch", lambda: _torch())
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: None)

    row = doctor.check_gpu(ctx)

    assert row.status == WARN
    assert "slow" in row.message


def test_an_nvidia_card_with_a_cpu_only_torch_still_fails(ctx, monkeypatch):
    """A driver with no CUDA build is a real, fixable fault.

    The new branch must not swallow it: this machine HAS a card spaCR
    could use and a torch that never will, and the fix is a reinstall.
    """
    monkeypatch.setattr(doctor, "_import_torch", lambda: _torch())
    monkeypatch.setattr(doctor, "_nvidia_driver", lambda: "580.173.02")

    row = doctor.check_gpu(ctx)

    assert row.status == FAIL
    assert "CPU-only build" in row.message
