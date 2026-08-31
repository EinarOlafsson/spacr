"""One resolver, and a fake probe per backend.

None of these backends can be tested for real: no machine has all five, and
the machine this was written on has exactly one of them. So the FAKE IS THE
TEST, and each of these must fail when its branch is deleted -- that is the
only thing standing between a backend and a silent regression to the CPU.

The distinction under most of this file is the one instruction 319 is about:
``torch.cuda.is_available()`` answers "is there CUDA", and about a dozen
spaCR call sites use it to mean "is there a GPU". ROCm makes those two
answers disagree -- it says True and is not NVIDIA -- and Metal makes them
disagree the other way. So :func:`is_gpu` and :func:`is_cuda` are tested
apart, on the same machines, in both directions.
"""
from __future__ import annotations

import pytest

from spacr import accelerator as acc


@pytest.fixture(autouse=True)
def _no_cached_answer(monkeypatch):
    """The resolver caches; every test here starts from an unprobed state."""
    monkeypatch.setattr(acc, "_CACHED", None, raising=False)
    monkeypatch.delenv(acc.ENV_DEVICE, raising=False)
    yield
    monkeypatch.setattr(acc, "_CACHED", None, raising=False)


class _Backends:
    """A stand-in for the ``torch.backends`` namespace."""

    def __init__(self, mps_available=False, mps_built=False):
        self.mps = _MPS(mps_available, mps_built)


class _MPS:
    def __init__(self, available, built):
        self._available, self._built = available, built

    def is_available(self):
        return self._available

    def is_built(self):
        return self._built


class _Cuda:
    def __init__(self, available, name="GPU"):
        self._available, self._name = available, name

    def is_available(self):
        return self._available

    def get_device_name(self, _index=0):
        return self._name

    def empty_cache(self):
        pass


class _Version:
    def __init__(self, cuda=None, hip=None):
        self.cuda, self.hip = cuda, hip


class _Torch:
    """Only the surface the resolver actually probes."""

    def __init__(self, cuda=False, cuda_name="GPU", cuda_version=None,
                 hip=None, mps=False, mps_built=False, xpu=None):
        self.cuda = _Cuda(cuda, cuda_name)
        self.version = _Version(cuda_version, hip)
        self.backends = _Backends(mps, mps_built)
        if xpu is not None:
            self.xpu = xpu

    @staticmethod
    def device(spec):
        return f"device({spec})"


def _install(monkeypatch, torch, directml=None):
    monkeypatch.setattr(acc, "_torch", lambda: torch)
    monkeypatch.setattr(acc, "_directml", lambda: directml)
    monkeypatch.setattr(acc, "_metal_gpu_name", lambda: "Test Metal GPU")


# ---------------------------------------------------------------------------
# One test per backend: the device string is the contract
# ---------------------------------------------------------------------------

def test_nvidia_is_unchanged(monkeypatch):
    """THE CONFIGURATION EVERY CURRENT USER HAS.

    Asserted explicitly and first, because the whole risk of this change is
    that a machine which already worked stops working. CUDA must still win,
    still be called cuda:0, and still be allowed double precision and
    autocast.
    """
    _install(monkeypatch, _Torch(cuda=True, cuda_name="RTX 4090",
                                 cuda_version="12.1"))
    found = acc.resolve(refresh=True)
    assert found.kind == "cuda"
    assert found.device == "cuda:0"
    assert found.is_cuda and found.is_gpu
    assert found.float64 and found.autocast
    assert "RTX 4090" in found.label and "NVIDIA" in found.label


def test_rocm_is_a_gpu_but_is_not_cuda(monkeypatch):
    """The trap that makes AMD-on-Linux report as absent.

    ROCm answers ``torch.cuda.is_available()`` True and takes
    ``device="cuda"``, so a site asking "is there a GPU" must say yes while
    a site asking "is this NVIDIA" must say no. Both, on the same machine.
    """
    _install(monkeypatch, _Torch(cuda=True, cuda_name="Radeon RX 7900 XTX",
                                 hip="6.0"))
    found = acc.resolve(refresh=True)
    assert found.kind == "rocm"
    assert found.device == "cuda:0", "ROCm dispatches through the cuda device"
    assert found.is_gpu is True
    assert found.is_cuda is False, "reporting ROCm as CUDA names the wrong vendor"
    assert "AMD" in found.label and "NVIDIA" not in found.label


def test_metal_is_selected_and_declares_what_it_cannot_do(monkeypatch):
    """Metal, which drives Apple Silicon AND AMD cards on Intel Macs.

    The capability flags are not decoration: float64 raises TypeError on
    this backend and autocast raises RuntimeError, both measured. A
    resolver that returned only a device string would hand those failures
    to a training run.
    """
    _install(monkeypatch, _Torch(cuda=False, mps=True, mps_built=True))
    found = acc.resolve(refresh=True)
    assert found.kind == "mps"
    assert found.device == "mps"
    assert found.is_gpu is True
    assert found.is_cuda is False
    assert found.float64 is False, "float64 raises TypeError on MPS"
    assert found.autocast is False, "autocast raises RuntimeError on MPS"
    assert found.fallback is True


def test_metal_built_but_unavailable_is_found_not_used(monkeypatch):
    """The state that would otherwise read as a broken install.

    torch ships Metal support on every mac build, so ``is_built()`` is True
    on machines that have no Metal device at all. That is DETECTED and NOT
    USABLE -- the two facts 319 insists on keeping apart.
    """
    _install(monkeypatch, _Torch(cuda=False, mps=False, mps_built=True))
    found = acc.resolve(refresh=True)
    assert found.detected is True
    assert found.usable is False
    assert found.is_gpu is False
    assert found.device == "cpu"
    assert found.note, "an unusable accelerator must say why"


def test_intel_xpu_is_selected(monkeypatch):
    class _Xpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_index=0):
            return "Arc A770"

    _install(monkeypatch, _Torch(cuda=False, xpu=_Xpu()))
    found = acc.resolve(refresh=True)
    assert found.kind == "xpu"
    assert found.device == "xpu"
    assert found.is_gpu is True
    assert "Arc A770" in found.label


def test_directml_is_selected(monkeypatch):
    _install(monkeypatch, _Torch(cuda=False),
             directml=acc.Accelerator(kind="directml", device="privateuseone:0",
                                      label="Radeon 780M (DirectML)",
                                      float64=False, autocast=False))
    found = acc.resolve(refresh=True)
    assert found.kind == "directml"
    assert found.device == "privateuseone:0"
    assert found.is_gpu is True


def test_a_plain_cpu_machine_resolves_to_cpu(monkeypatch):
    _install(monkeypatch, _Torch(cuda=False))
    found = acc.resolve(refresh=True)
    assert found.kind == "cpu"
    assert found.device == "cpu"
    assert found.is_gpu is False


# ---------------------------------------------------------------------------
# Preference order, and the promise that nothing here raises
# ---------------------------------------------------------------------------

def test_cuda_wins_over_everything_else(monkeypatch):
    """A machine with both must not be moved off the backend it was tuned on."""
    class _Xpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_index=0):
            return "Arc"

    _install(monkeypatch, _Torch(cuda=True, cuda_version="12.1",
                                 mps=True, mps_built=True, xpu=_Xpu()))
    assert acc.resolve(refresh=True).kind == "cuda"


def test_a_half_installed_backend_falls_back_instead_of_raising(monkeypatch):
    """A machine with a broken ROCm must still start.

    The accelerators most likely to be half-installed are on the machines
    least able to afford a traceback at import time, so a probe that throws
    while being ASKED whether it exists resolves to the CPU.
    """
    class _Exploding:
        @staticmethod
        def is_available():
            raise RuntimeError("libamdhip64.so: cannot open shared object file")

        @staticmethod
        def get_device_name(_index=0):
            raise RuntimeError("no")

    torch = _Torch(cuda=False)
    torch.cuda = _Exploding()
    _install(monkeypatch, torch)
    found = acc.resolve(refresh=True)
    assert found.kind == "cpu"
    assert found.is_gpu is False


def test_no_torch_at_all_is_the_cpu_and_says_so(monkeypatch):
    monkeypatch.setattr(acc, "_torch", lambda: None)
    found = acc.resolve(refresh=True)
    assert found.kind == "cpu"
    assert "PyTorch" in found.note


def test_the_environment_can_force_the_cpu(monkeypatch):
    """The escape hatch for "the GPU answer looks wrong"."""
    _install(monkeypatch, _Torch(cuda=True, cuda_version="12.1"))
    monkeypatch.setenv(acc.ENV_DEVICE, "cpu")
    found = acc.resolve(refresh=True)
    assert found.kind == "cpu"
    assert found.is_gpu is False


def test_the_environment_can_force_a_device_detection_did_not_pick(monkeypatch):
    _install(monkeypatch, _Torch(cuda=False))
    monkeypatch.setenv(acc.ENV_DEVICE, "mps")
    found = acc.resolve(refresh=True)
    assert found.device == "mps"
    assert found.float64 is False, "a forced Metal device still cannot do float64"


# ---------------------------------------------------------------------------
# What the call sites ask
# ---------------------------------------------------------------------------

def test_cellpose_is_told_yes_on_every_gpu_not_only_cuda(monkeypatch):
    """The single flag that pinned every Mac to the CPU.

    Cellpose resolves its own device and already knows about Metal, but it
    branches on ``gpu=`` BEFORE it looks at ``device=``. Passing
    ``torch.cuda.is_available()`` there is what made a working Radeon sit
    idle, so this asserts the flag follows "is there a GPU", not "is there
    CUDA".
    """
    _install(monkeypatch, _Torch(cuda=False, mps=True, mps_built=True))
    assert acc.cellpose_gpu() is True

    monkeypatch.setattr(acc, "_CACHED", None, raising=False)
    _install(monkeypatch, _Torch(cuda=True, hip="6.0"))
    assert acc.cellpose_gpu() is True

    monkeypatch.setattr(acc, "_CACHED", None, raising=False)
    _install(monkeypatch, _Torch(cuda=False))
    assert acc.cellpose_gpu() is False


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(cuda=True, cuda_version="12.1"), "cuda"),
        (dict(cuda=True, hip="6.0"), "cuda"),
        (dict(cuda=False, mps=True, mps_built=True), None),
        (dict(cuda=False), None),
    ],
)
def test_autocast_device_type_is_none_where_autocast_raises(
        monkeypatch, kwargs, expected):
    """None means "do not use mixed precision", not "use it on the CPU".

    Those are different instructions and the difference has to survive to
    the call site: ``torch.autocast(device_type="mps")`` raises outright,
    so deep_spacr has to skip the context manager rather than re-point it.
    """
    _install(monkeypatch, _Torch(**kwargs))
    assert acc.autocast_device_type() == expected


def test_empty_cache_is_safe_on_every_backend(monkeypatch):
    """Called from resource cleanup, which must not be able to raise."""
    for kwargs in (dict(cuda=True, cuda_version="12.1"),
                   dict(cuda=False, mps=True, mps_built=True),
                   dict(cuda=False)):
        monkeypatch.setattr(acc, "_CACHED", None, raising=False)
        _install(monkeypatch, _Torch(**kwargs))
        acc.resolve(refresh=True)
        acc.empty_cache()


def test_describe_separates_found_from_in_use(monkeypatch):
    """What the setup slide and the doctor print.

    A slide announcing an accelerator spaCR will not dispatch to makes the
    user blame their hardware for CPU speed, so "found, not used" has to be
    a distinct sentence from "in use".
    """
    _install(monkeypatch, _Torch(cuda=False, mps=True, mps_built=True))
    assert "in use" in acc.describe()

    monkeypatch.setattr(acc, "_CACHED", None, raising=False)
    _install(monkeypatch, _Torch(cuda=False, mps=False, mps_built=True))
    text = acc.describe()
    assert "found" in text and "not used" in text
