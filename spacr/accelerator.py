"""Select the torch compute device and report its vendor name and limitations.

About forty-five spaCR call sites ask ``torch.cuda.is_available()``, and
most of them use it to decide whether any GPU is present. On non-NVIDIA
hardware those are two different questions, and treating them as one
reports a usable accelerator as "no GPU":

* ROCm answers ``torch.cuda.is_available()`` **True** and wants
  ``device="cuda"``. Code that then prints "CUDA" names the wrong vendor;
  code that prints "no NVIDIA GPU" hides a usable AMD device.
* Apple's Metal answers it **False** and wants ``device="mps"``. MPS is
  *Metal*, not an Apple-Silicon feature. On
  the Intel iMac this was written on it drives an AMD Radeon Pro 5300 --
  measured 16x on conv2d inference and 2.7x on a training step against
  the same machine's CPU. ROCm has no macOS build, so Metal is the only
  route to an AMD card on a Mac.

This module therefore reports the vendor and the GPU decision separately,
and also records what the selected device cannot do. Moving a tensor to
MPS is not sufficient, because three operations fail rather than slow
down (each was measured):

* ``float64`` raises ``TypeError``. Computations that need double
  precision must stay on the CPU.
* ``autocast(device_type="mps")`` raises ``RuntimeError``. Mixed precision
  must be disabled on this device, not redirected to it.
* Some operators are missing (``aten::linalg_qr.out`` among them).
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` makes those run on the CPU instead of
  raising, so this module sets it when it selects MPS.

Detected and usable hardware are reported separately. A setup screen that
lists an accelerator spaCR never dispatches to leads users to attribute CPU
run times to that hardware. Neural engines are the main case: the Apple
Neural Engine, Intel AI Boost and Qualcomm Hexagon have no portable torch
device and are never selected; :func:`neural_engines` currently reports
only the Apple Neural Engine, on arm64 macOS.

Device resolution does not raise. A half-installed ROCm, a broken driver,
or a torch build without a backend it advertises all resolve to the CPU
with an explanatory note, so an unusual accelerator cannot cause a
traceback at import time.
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Optional, Tuple

LOG = logging.getLogger("spacr.accelerator")

#: Force a device, or force the CPU. The escape hatch for "the GPU answer
#: looks wrong" that does not require uninstalling anything -- the same
#: shape as ``SPACR_USE_RAPIDS`` in :mod:`spacr.gpu_reduce`.
ENV_DEVICE = "SPACR_DEVICE"

#: Kinds in the order they are preferred. CUDA first because it is what
#: every current user has and what the numbers in this project were taken
#: on; CPU last because it always answers.
KINDS: Tuple[str, ...] = ("cuda", "rocm", "mps", "xpu", "directml", "cpu")


@dataclass(frozen=True)
class Accelerator:
    """Resolved compute device, its display names and its capabilities.

    :param kind: one of :data:`KINDS`. ``rocm`` is reported distinctly from
        ``cuda`` even though both dispatch to ``device="cuda"``, because
        the only thing a user can act on is the true vendor.
    :param device: the string to hand ``torch.device``.
    :param label: human text for the setup slide and the doctor.
    :param name: undecorated hardware marketing name shown to users, without
        the backend or version suffix carried by ``label``; empty when no
        device name was discovered.
    :param detected: the hardware is present.
    :param usable: spaCR dispatches work to it. Never true unless
        ``detected``; false for accelerators with no torch device.
    :param note: why it is not usable, when it is not. Empty otherwise.
    :param float64: the device accepts double precision.
    :param autocast: ``torch.autocast`` accepts this device type.
    :param fallback: missing operators silently run on the CPU instead of
        raising, because this module set the backend's fallback flag.
    :param bfloat16: the device accepts ``torch.bfloat16``. Probed rather
        than assumed from the backend name, because it is the one
        capability that moves with the torch version: Metal gained it
        after 2.2, and 2.2.2 is the last x86-64 macOS wheel, so the same
        backend answers differently on an Intel Mac and an Apple Silicon
        one. Cellpose's cpsam loads its weights in bfloat16 by default,
        so a wrong answer here is a TypeError at model construction.
    """

    kind: str
    device: str
    label: str
    #: The card's own name, undecorated -- "AMD Radeon Pro 5300", not
    #: "AMD Radeon Pro 5300 (Apple Metal)". SEPARATE FROM `label` because
    #: the two have different jobs: a user recognises the name from the box
    #: and from About This Mac, while the route only matters where it is
    #: surprising. On NVIDIA the two would differ only by a suffix that
    #: repeats the word NVIDIA, so the setup slide shows this undecorated name.
    name: str = ""
    detected: bool = True
    usable: bool = True
    note: str = ""
    float64: bool = True
    autocast: bool = True
    fallback: bool = False
    bfloat16: bool = True

    @property
    def is_gpu(self) -> bool:
        """Whether a usable non-CPU device was selected.

        This is the check most call sites intend when they write
        ``torch.cuda.is_available()``.
        """
        return self.usable and self.kind != "cpu"

    @property
    def is_cuda(self) -> bool:
        """Whether this accelerator is NVIDIA CUDA (not ROCm).

        Narrower than :attr:`is_gpu`: anything reading
        ``nvidia-smi``, a CUDA version, or NVIDIA-specific memory
        interfaces wants this one and would be wrong on ROCm.
        """
        return self.kind == "cuda"

    @property
    def torch_device(self):
        """A ``torch.device`` for :attr:`device`; requires torch to be
        installed."""
        import torch

        return torch.device(self.device)


_CPU = Accelerator(
    kind="cpu", device="cpu", label="CPU", detected=True, usable=True,
    note="", float64=True, autocast=True, fallback=False)

#: Resolved once. Probing costs a CUDA context on some drivers, and the
#: answer cannot change inside a process.
_CACHED: Optional[Accelerator] = None


def _reset_device_cache() -> None:
    """Forget the cached accelerator so the next :func:`resolve` probes again.

    For test isolation. A test that fakes the machine -- ``torch.cuda``
    patched to answer False, say -- is only believed if nothing was cached
    before the patch; otherwise the real ``cuda:0`` answers for the fake
    CPU-only machine and ``torch.load(map_location=...)`` sends the weights
    to a GPU the test said was not there. Imports nothing and probes
    nothing, so clearing is free; the re-probe is paid only by a caller
    that actually asks.
    """
    global _CACHED
    _CACHED = None


def _torch():
    """torch, or None. spaCR runs without it for plenty of tasks."""
    try:
        import torch

        return torch
    except Exception:                                        # noqa: BLE001
        return None


def _forced() -> Optional[str]:
    """The device named in the environment, normalised, or None."""
    wanted = str(os.environ.get(ENV_DEVICE, "")).strip().lower()
    return wanted or None


def _cuda_or_rocm(torch, *, device_names=True) -> Optional[Accelerator]:
    """NVIDIA, or AMD-on-Linux, both of which dispatch to ``cuda``."""
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:                                        # noqa: BLE001
        return None
    version_module = getattr(torch, "version", None)
    hip = getattr(version_module, "hip", None)
    try:
        name = torch.cuda.get_device_name(0) if device_names else "GPU"
    except Exception:                                        # noqa: BLE001
        name = "GPU"
    if hip:
        return Accelerator(
            kind="rocm", device="cuda:0",
            label=f"{name} (AMD ROCm {hip})", name=name,
            float64=True, autocast=True)
    version = getattr(version_module, "cuda", None)
    label = f"{name} (NVIDIA CUDA {version})" if version else name
    return Accelerator(kind="cuda", device="cuda:0", label=label, name=name)


def _mps(torch) -> Optional[Accelerator]:
    """Apple Metal. Drives Apple Silicon AND AMD cards on Intel Macs."""
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is None:
        return None
    try:
        if not backend.is_available():
            if backend.is_built():
                label, note = _why_metal_is_unavailable()
                return Accelerator(
                    kind="mps", device="cpu", label=label,
                    name=label, detected=True, usable=False, note=note)
            return None
    except Exception:                                        # noqa: BLE001
        return None
    metal_name = _metal_gpu_name()
    return Accelerator(
        kind="mps", device="mps", label=f"{metal_name} (Apple Metal)",
        name=metal_name,
        float64=False, autocast=False, fallback=True)


#: Intel's integrated GPU families, as `system_profiler` spells them.
#:
#: These are the parts in Intel Macs that have no PyTorch backend AT ALL --
#: see :func:`_why_metal_is_unavailable`. Matched on the name because that is
#: the only thing available: torch reports nothing about a device it refuses
#: to open, and `system_profiler` is already being asked for the marketing
#: name a page or two above.
_INTEL_INTEGRATED = ("iris", "uhd graphics", "hd graphics")


def _why_metal_is_unavailable() -> Tuple[str, str]:
    """``(label, note)`` for a torch with Metal built in and no Metal device.

    THREE DIFFERENT ANSWERS, because the reader can act on them differently
    and the third one is the whole reason this function exists.

    * macOS older than 12.3: the MPS backend did not exist yet. Upgrading
      fixes it, and that is worth saying.
    * An Intel Mac with Intel integrated graphics: NOTHING fixes it, and
      saying so is the useful answer. Apple's Metal backend for PyTorch
      covers Apple Silicon and the AMD discrete cards Apple shipped in
      Intel Macs; Intel's integrated parts were never added. Intel's own
      PyTorch path -- IPEX, ``torch.xpu`` -- targets the Arc and Xe
      DISCRETE GPUs on Linux and Windows, has no macOS build, and does not
      support these iGPUs either. So the CPU is not a fallback here, it is
      the only device, and a user who goes looking for the driver they must
      have installed wrong will not find one.
    * Anything else: the honest generic, which is what this function
      answered for every case before.

    :returns: the label the doctor puts in front of "was detected but spaCR
        cannot use it", and the sentence under it.
    """
    if platform.system() != "Darwin":
        return ("Apple Metal (MPS)",
                "torch is built with Metal support but this is not macOS, "
                "so there is no Metal device to use; running on the CPU.")

    gpu = _metal_gpu_name()
    release = platform.mac_ver()[0]
    intel_mac = (platform.machine() in ("x86_64", "i386")
                 and not _rosetta_translated())

    if release:
        try:
            major, minor = (int(part) for part in release.split(".")[:2])
        except ValueError:                                   # noqa: BLE001
            major = minor = 0
        if major and (major, minor) < (12, 3):
            return (gpu,
                    f"macOS {release} is older than 12.3, which is where "
                    f"PyTorch's Metal (MPS) backend begins. Upgrading macOS "
                    f"is what enables it; until then spaCR runs on the CPU.")

    lowered = gpu.lower()
    if intel_mac and any(family in lowered for family in _INTEL_INTEGRATED):
        return (gpu,
                f"{gpu} is Intel integrated graphics in an Intel Mac, and no "
                f"PyTorch backend supports it. Apple's Metal (MPS) backend "
                f"covers Apple Silicon and the AMD discrete cards Apple "
                f"shipped in Intel Macs; Intel's integrated GPUs were never "
                f"added to it. Intel's own PyTorch path -- IPEX and "
                f"torch.xpu -- targets the Arc and Xe discrete cards on "
                f"Linux and Windows, has no macOS build, and does not "
                f"support this GPU either. spaCR runs on the CPU here and "
                f"THERE IS NOTHING TO INSTALL OR CONFIGURE: this is a gap "
                f"between two vendors, not a setup problem. Segmentation "
                f"and training will be slow here, so run the batch on a "
                f"CUDA machine instead: `spacr-remote` is built for that "
                f"and shares its profiles with the Distributed Jobs screen, "
                f"which leaves this Mac as the client it is good at being.")

    return (gpu,
            "torch has Metal support but this system does not offer a Metal "
            "device to it; running on the CPU.")


@lru_cache(maxsize=1)
def _metal_gpu_name() -> str:
    """The GPU's marketing name on macOS, or a safe generic.

    Worth the subprocess exactly once: "AMD Radeon Pro 5300" is what the
    user sees on the box and in About This Mac, and a setup slide that
    says "MPS device" instead has told them nothing they can check.
    """
    if platform.system() != "Darwin":
        return "Metal GPU"
    if _apple_silicon():
        chip = _sysctl_string(b"machdep.cpu.brand_string")
        if chip.startswith("Apple "):
            return chip
    try:
        import subprocess

        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "Chipset Model:" in line:
                return line.split(":", 1)[1].strip()
    except Exception:                                        # noqa: BLE001
        pass
    return "Metal GPU"


@lru_cache(maxsize=1)
def _rosetta_translated() -> bool:
    """True when this process is Intel code running under Rosetta 2.

    ``platform.machine()`` cannot answer this: a translated process is told
    ``x86_64``, so an M2 running an Intel Anaconda looks exactly like an
    Intel Mac to every other check in spaCR. An environment made with
    ``conda create -n spacr`` from an Intel Anaconda on Apple Silicon puts
    spaCR on an emulated Python with torch capped at 2.2.2 (the last
    Intel-Mac wheel), and without this check nothing in spaCR says so.

    The kernel answers it directly through ``sysctl.proc_translated``: 1
    when translated, 0 when native, and absent on an Intel Mac. Read
    through ``sysctlbyname`` rather than a ``sysctl`` subprocess because
    it is asked at launch. Never raises.
    """
    if platform.system() != "Darwin":
        return False
    try:
        import ctypes

        libc = _libc()
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(b"sysctl.proc_translated", ctypes.byref(value),
                             ctypes.byref(size), None, ctypes.c_size_t(0)):
            return False
        return value.value == 1
    except Exception:                                        # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def _libc():
    """The C library, for ``sysctlbyname``.

    ``CDLL(None)`` is the process's own symbol table, which on macOS already
    holds libSystem; ``find_library("c")`` scans dyld paths for ~30 ms first.
    """
    import ctypes

    try:
        libc = ctypes.CDLL(None)
        libc.sysctlbyname
        return libc
    except (OSError, AttributeError):
        import ctypes.util

        return ctypes.CDLL(ctypes.util.find_library("c"))


def _sysctl_string(name: bytes) -> str:
    """A string sysctl read through ``sysctlbyname``, or ``""``. Never raises."""
    try:
        import ctypes

        libc = _libc()
        size = ctypes.c_size_t(0)
        if libc.sysctlbyname(name, None, ctypes.byref(size), None, ctypes.c_size_t(0)):
            return ""
        buffer = ctypes.create_string_buffer(size.value)
        if libc.sysctlbyname(name, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)):
            return ""
        return buffer.value.decode("utf-8", "replace").strip()
    except Exception:                                        # noqa: BLE001
        return ""


def _apple_silicon() -> bool:
    """True on an Apple Silicon Mac, whether or not this Python is native."""
    return (platform.system() == "Darwin"
            and (platform.machine() == "arm64" or _rosetta_translated()))


#: What to do about :func:`_rosetta_translated`. ``CONDA_SUBDIR`` is the part
#: that matters: an Intel Anaconda creates Intel environments by default, so
#: the plain ``conda create`` the doctor used to suggest rebuilt the problem.
#: Verified on that M2 with Anaconda's x86_64 conda 23.7.4.
_ROSETTA_FIX = (
    "Create a native arm64 environment and reinstall spaCR into it:\n"
    "CONDA_SUBDIR=osx-arm64 conda create -n spacr-arm64 python=3.12 -y && "
    "conda activate spacr-arm64 && conda config --env --set subdir osx-arm64 && "
    'python -m pip install "spacr"\n'
    "(or install Miniforge for Apple Silicon, or the spaCR macOS installer, "
    "which installs a native runtime itself)"
)


def _xpu(torch, *, device_names=True) -> Optional[Accelerator]:
    """Intel Arc / Xe, on a torch built with XPU or with IPEX loaded."""
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return None
    try:
        if not xpu.is_available():
            return None
    except Exception:                                        # noqa: BLE001
        return None
    try:
        name = xpu.get_device_name(0) if device_names else "Intel GPU"
    except Exception:                                        # noqa: BLE001
        name = "Intel GPU"
    return Accelerator(kind="xpu", device="xpu", label=f"{name} (Intel XPU)",
                       name=name, float64=False, autocast=False,
                       fallback=True)


def _directml() -> Optional[Accelerator]:
    """Windows DirectML: any vendor, thinnest operator coverage."""
    try:
        import torch_directml
    except Exception:                                        # noqa: BLE001
        return None
    try:
        if not torch_directml.is_available():
            return None
        device = str(torch_directml.device())
        name = torch_directml.device_name(0)
    except Exception:                                        # noqa: BLE001
        return None
    return Accelerator(kind="directml", device=device,
                       label=f"{name} (DirectML)", name=name,
                       float64=False, autocast=False, fallback=True)


def neural_engines() -> Tuple[str, ...]:
    """Names of detected neural-inference hardware that spaCR does not use.

    Reported so the setup slide can say "found, not used" for an idle
    Neural Engine. There is no portable torch device for any of these --
    the ANE is reachable only through CoreML, Intel's AI Boost only
    through OpenVINO -- so none of them is offered as a compute device.
    """
    found = []
    if _apple_silicon():
        found.append("Apple Neural Engine (CoreML only)")
    return tuple(found)


def inspect_torch(torch, *, device_names=True, include_cuda=True) -> Accelerator:
    """Resolve the accelerator for a given torch module, bypassing the cache.

    For callers that already hold a torch handle and need the result for
    that handle -- :func:`spacr.doctor.check_gpu` imports torch through its
    own indirection so the diagnosis can be tested against a stand-in, and
    the cached result for the real machine would ignore that stand-in. Uses
    the same probes in the same order as :func:`resolve`.

    :param torch: torch module whose availability metadata should be inspected.
    :param device_names: False avoids CUDA/XPU property queries that can
        initialize their runtime. Availability queries still run.
    :param include_cuda: False skips CUDA/ROCm entirely, for a process whose
        environment deliberately hides those devices. Other backends remain
        eligible. This function does not allocate dtype-probe tensors.
    :returns: accelerator described by the supplied torch module.
    """
    found = None
    for probe in (lambda: _cuda_or_rocm(torch, device_names=device_names) if include_cuda else None,
                  lambda: _mps(torch), lambda: _xpu(torch, device_names=device_names), _directml):
        try:
            found = probe()
        except Exception:                                    # noqa: BLE001
            found = None
        if found is not None:
            return found
    return _CPU


def resolve(refresh: bool = False) -> Accelerator:
    """The accelerator spaCR will use. Cached; never raises.

    :param refresh: probe again instead of answering from the cache. For
        tests, which fake the probes.
    """
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    torch = _torch()
    forced = _forced()
    if forced in ("cpu", "none", "0", "off"):
        _CACHED = Accelerator(kind="cpu", device="cpu", label="CPU",
                              note=f"forced by {ENV_DEVICE}")
        return _CACHED
    if torch is None:
        _CACHED = Accelerator(kind="cpu", device="cpu", label="CPU",
                              note="PyTorch is not installed")
        return _CACHED

    found: Optional[Accelerator] = None
    for probe in (lambda: _cuda_or_rocm(torch), lambda: _mps(torch),
                  lambda: _xpu(torch), _directml):
        try:
            found = probe()
        except Exception:                                    # noqa: BLE001
            LOG.debug("accelerator probe failed", exc_info=True)
            found = None
        if found is not None:
            break

    if found is None:
        found = _CPU
    if forced and forced not in ("auto", ""):
        found = _forced_device(forced, found)
    if found.usable and found.kind != "cpu":
        found = _measure_dtypes(torch, found)
    if found.usable and found.fallback:
        found = replace(found, fallback=_mps_fallback_live(found))
    _CACHED = found
    LOG.debug("accelerator resolved to %s (%s)", found.device, found.label)
    return found


def _mps_fallback_live(found: Accelerator) -> bool:
    """Whether operators the device lacks will run on the CPU, warning when not.

    The variable alone is not the answer: torch reads it once, at import, so
    a notebook that ran ``import torch`` before ``import spacr`` has it set
    and no fallback. On an M2, ``torch.linalg.eig`` on Metal then raised
    NotImplementedError instead of running on the CPU.
    """
    if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""):
        return False
    import spacr

    if getattr(spacr, "_MPS_FALLBACK_LIVE", True):
        return True
    if found.kind == "mps":
        import warnings

        warnings.warn(
            "torch was imported before spaCR, so PYTORCH_ENABLE_MPS_FALLBACK "
            "is not active and operators Apple Metal lacks will raise "
            "NotImplementedError instead of running on the CPU. Import spacr "
            "before torch, or set PYTORCH_ENABLE_MPS_FALLBACK=1 before "
            "starting Python.", RuntimeWarning, stacklevel=3)
    return False


def _measure_dtypes(torch, found: Accelerator) -> Accelerator:
    """Ask the device what it accepts, rather than believing a table.

    A two-element zero tensor per dtype, on a device that already answered
    "available". Cheap, and it is the difference between a capability
    matrix that ages and one that cannot: every backend here gains dtypes
    across torch releases, and a hardcoded ``bfloat16=False`` for Metal
    would still be wrong on the next machine.

    A probe that cannot even allocate leaves the declared value alone --
    the backend is answering strangely and guessing further would be worse
    than the conservative default it already carries.
    """
    def accepts(name: str) -> Optional[bool]:
        """Probe whether the resolved device accepts one named torch dtype.

        :param name: torch dtype attribute name, such as ``"float64"`` or
            ``"bfloat16"``.
        :returns: ``True`` when a two-element tensor allocates on the device,
            ``False`` for the errors used by unsupported dtypes, or ``None``
            when torch lacks the dtype or the probe fails unexpectedly.
        """
        dtype = getattr(torch, name, None)
        if dtype is None:
            return None
        try:
            torch.zeros(2, dtype=dtype, device=found.device)
            return True
        except (TypeError, RuntimeError):
            return False
        except Exception:                                    # noqa: BLE001
            return None

    measured = {}
    for flag, name in (("float64", "float64"), ("bfloat16", "bfloat16")):
        answer = accepts(name)
        if answer is not None:
            measured[flag] = answer
    return replace(found, **measured) if measured else found


def _forced_device(wanted: str, found: Accelerator) -> Accelerator:
    """Honour ``SPACR_DEVICE`` even when it names something unprobed.

    Deliberately permissive: someone setting this is overriding detection
    on purpose, usually because detection is what is wrong.
    """
    if wanted in (found.device, found.kind):
        return found
    return Accelerator(kind=wanted.split(":")[0], device=wanted,
                       label=f"{wanted} (forced by {ENV_DEVICE})",
                       float64=not wanted.startswith(("mps", "xpu")),
                       autocast=not wanted.startswith(("mps", "xpu")),
                       fallback=wanted.startswith("mps"))



def _mask_devices():
    """List CUDA/ROCm mask devices visible to this process, without tensors.

    Logical indices belong to the current allocation, including scheduler
    visibility restrictions. Other accelerator types are not multi-GPU mask
    candidates. An explicit single-device override disables parallel discovery.
    Unavailable or failed probes return an empty tuple.
    """
    forced = _forced()
    if forced not in (None, 'auto', 'cuda', 'rocm'):
        return ()
    torch = _torch()
    if torch is None:
        return ()
    try:
        if not torch.cuda.is_available():
            return ()
        backend = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
        if forced in ('cuda', 'rocm') and forced != backend:
            return ()
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({'index': index, 'name': str(properties.name),
                            'memory_bytes': int(properties.total_memory),
                            'backend': backend})
        return tuple(devices)
    except Exception:
        LOG.debug('Could not enumerate mask GPUs', exc_info=True)
        return ()


def _mask_worker_environment(index, *, backend, count, environment=None):
    """Restrict a fresh mask worker to one GPU from its parent's allocation.

    CUDA visibility lists remap logical ordinals. HIP uses the same mapping
    within any retained ROCR restriction. Never replace an allocated logical
    ordinal with a host ordinal or remove a scheduler's ROCR restriction.
    The returned environment is independent; this process is not modified.

    :param index: selected logical device ordinal reported by _mask_devices.
    :param backend: ``cuda`` or ``rocm`` from device discovery.
    :param count: number of devices the parent actually discovered.
    :param environment: parent environment, defaulting to os.environ.
    :returns: child environment exposing the selected device as cuda:0.
    :raises ValueError: for an invalid selection or contradictory HIP aliases.
    """
    import operator

    try:
        if isinstance(index, bool) or isinstance(count, bool):
            raise TypeError
        index, count = operator.index(index), operator.index(count)
    except TypeError as exc:
        raise ValueError('GPU selection and device count must be integers') from exc
    if backend not in ('cuda', 'rocm') or not 0 <= index < count:
        raise ValueError('Selected GPU is outside the available mask devices')
    result = dict(os.environ if environment is None else environment)
    key = 'CUDA_VISIBLE_DEVICES'
    if backend == 'rocm' and 'HIP_VISIBLE_DEVICES' in result:
        key = 'HIP_VISIBLE_DEVICES'
        if ('CUDA_VISIBLE_DEVICES' in result
                and result['CUDA_VISIBLE_DEVICES'] != result[key]):
            raise ValueError('HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES disagree')
    if key in result:
        visible = [token.strip() for token in result[key].split(',')]
        if len(visible) < count or not visible[index] or visible[index] == '-1':
            raise ValueError('GPU visibility changed since device discovery')
        selected = visible[index]
    else:
        selected = str(index)
    result['CUDA_VISIBLE_DEVICES'] = selected
    if backend == 'rocm':
        result['HIP_VISIBLE_DEVICES'] = selected
    result[ENV_DEVICE] = backend
    return result


def torch_device():
    """``torch.device`` for the resolved accelerator.

    The direct replacement for
    ``torch.device("cuda:0" if torch.cuda.is_available() else "cpu")``.
    """
    return resolve().torch_device


def device_string() -> str:
    """The resolved device as a string, e.g. ``"cuda:0"``, ``"mps"``."""
    return resolve().device


def is_gpu() -> bool:
    """Whether a usable accelerator of any vendor was selected.

    What a call site means when it writes ``torch.cuda.is_available()`` to
    decide whether to use a GPU at all.
    """
    return resolve().is_gpu


def is_cuda() -> bool:
    """Whether the resolved accelerator is NVIDIA CUDA (not ROCm).

    For the places that need CUDA specifically -- memory
    interrogation, ``nvidia-smi`` advice, CUDA version reporting.
    """
    return resolve().is_cuda


def supports_float64() -> bool:
    """Whether double precision may be sent to the device."""
    return resolve().float64


def supports_autocast() -> bool:
    """Whether a GPU is selected and ``torch.autocast`` accepts its device type."""
    accelerator = resolve()
    return accelerator.is_gpu and accelerator.autocast


def autocast_device_type() -> Optional[str]:
    """The string for ``torch.autocast(device_type=...)``, or None.

    None means "do not use mixed precision here", which is a different
    answer from "use it on the CPU" and has to stay distinguishable.
    """
    accelerator = resolve()
    if not supports_autocast():
        return None
    return "cuda" if accelerator.kind in ("cuda", "rocm") else accelerator.kind


def cellpose_gpu() -> bool:
    """Return the value to pass Cellpose as ``gpu=``.

    Cellpose resolves the device itself and supports MPS --
    ``assign_device(gpu=True)`` returns ``mps`` on a Metal machine -- but it
    uses a GPU only when called with ``gpu=True``. Passing
    ``torch.cuda.is_available()`` here pins a Mac to the CPU when no explicit
    device is supplied. An explicit Cellpose device takes precedence.
    """
    return is_gpu()


def supports_bfloat16() -> bool:
    """Whether ``torch.bfloat16`` tensors may be put on the device."""
    return resolve().bfloat16


def cellpose_kwargs() -> dict:
    """Consistent device and weight precision for ``CellposeModel`` inference.

    The three arguments must agree, so they are produced together here
    rather than written out at six call sites:

    * ``gpu`` -- controls Cellpose's selection when no explicit device is
      supplied, including callers that deliberately drop ``device`` below.
    * ``device`` -- from the one resolver, so cellpose and spaCR cannot
      disagree about the same machine; an explicit device takes precedence.
    * ``use_bfloat16`` -- cpsam loads its weights in bfloat16 by default
      and Metal on torch 2.2 has no bfloat16, so the default is a
      ``TypeError: BFloat16 is not supported on MPS`` at construction.
      Measured on the reporting iMac; float32 weights work there at the cost
      of additional VRAM.
      CPU inference also uses float32: bfloat16 support does not imply
      native arithmetic, and emulation can be substantially slower. This
      matches Make Masks; float32 arithmetic need not produce identical
      predictions to bfloat16.

      Metal that accepts bfloat16 still gets float32 when the Mac has
      :data:`_METAL_FP32_MIN_MEMORY`: accepting the dtype is not computing
      in it. On an Apple M2 (torch 2.14, cellpose 4.2.1.1, 512x512) bfloat16
      weights took 7.85 s an image against 5.95 s in float32, the same 65
      masks (foreground IoU 0.9995), for 2.7 vs 4.1 GB of GPU memory. A
      smaller Mac keeps bfloat16, where that memory is the tighter limit.

    Callers that pass ``device=None`` on purpose -- letting cellpose
    resolve it -- should take ``gpu`` and ``use_bfloat16`` from here and
    drop ``device``.
    """
    accelerator = resolve()
    if accelerator.kind == "mps":
        _keep_cellpose_flows_off_metal()
    kwargs = {"gpu": accelerator.is_gpu, "device": accelerator.torch_device}
    if not accelerator.is_gpu or not accelerator.bfloat16:
        kwargs["use_bfloat16"] = False
    elif accelerator.kind == "mps" and _physical_memory() >= _METAL_FP32_MIN_MEMORY:
        kwargs["use_bfloat16"] = False
    return kwargs


#: Unified memory from which Metal runs Cellpose in float32; see
#: :func:`cellpose_kwargs`.
_METAL_FP32_MIN_MEMORY = 16 * 1024 ** 3


def _physical_memory() -> int:
    """Installed RAM in bytes, or 0 when the OS will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


#: Set once ``_keep_cellpose_flows_off_metal`` has patched the module.
_FLOW_PATCH_APPLIED = False


def _keep_cellpose_flows_off_metal() -> None:
    """Run cellpose's flow-error pass on the CPU when the device is Metal.

    A NARROW WORKAROUND FOR AN UPSTREAM BUG, and worth stating precisely
    because monkeypatching someone else's library needs a removal
    condition. ``cellpose.dynamics.masks_to_flows_gpu`` computes an index
    that lands one past the end on MPS::

        mu0[:, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu
        IndexError: index 256 is out of bounds for axis 1 with size 256

    Reproduced on a 256x256 image of ordinary round blobs, so it is not an
    artefact of a pathological input; the same image segments correctly on
    the CPU.

    WHAT IS AND IS NOT MOVED. Only ``remove_bad_flow_masks`` goes back to
    the CPU. It is post-processing over a finished mask, its own signature
    already DEFAULTS to ``device=torch.device("cpu")``, and it is cheap.
    The transformer encoder -- which is essentially all of the runtime;
    one 256x256 image took 444 s on this machine's CPU -- stays on Metal.

    TO REMOVE THIS: run the segmentation with a Metal device on a cellpose
    that has fixed the indexing, and delete the whole function. It is
    idempotent and never raises: a cellpose whose internals have moved
    leaves spaCR on the working path it already had.
    """
    global _FLOW_PATCH_APPLIED
    if _FLOW_PATCH_APPLIED:
        return
    _FLOW_PATCH_APPLIED = True
    try:
        import torch
        from cellpose import dynamics
    except Exception:                                        # noqa: BLE001
        return
    original = getattr(dynamics, "remove_bad_flow_masks", None)
    if original is None or getattr(original, "_spacr_cpu_flows", False):
        return

    def on_the_cpu(masks, flows, threshold=0.4, device=None):
        """Run Cellpose flow-error filtering on the CPU for Metal.

        :param masks: label masks forwarded unchanged to Cellpose.
        :param flows: flow field forwarded unchanged to Cellpose.
        :param threshold: flow-error threshold forwarded to Cellpose.
        :param device: ignored compatibility argument; the wrapped call
            always receives ``torch.device("cpu")``.
        :returns: the original ``remove_bad_flow_masks`` return value.
        """
        return original(masks, flows, threshold=threshold,
                        device=torch.device("cpu"))

    on_the_cpu._spacr_cpu_flows = True
    dynamics.remove_bad_flow_masks = on_the_cpu
    LOG.debug("cellpose flow-error pass pinned to the CPU (Metal indexing bug)")


def empty_cache(torch_module=None) -> str:
    """Release the backend's cached device memory and return the call made
    (``""`` when none was made). Never raises.

    :param torch_module: resolve the backend from this torch module instead
        of the cached result for the machine. :mod:`spacr.qt.resource_cleanup` holds its own
        handle -- it deliberately does not import torch just to free
        memory -- and a cached global would send the call to the wrong
        backend when that handle is a stand-in.
    """
    torch = torch_module if torch_module is not None else _torch()
    if torch is None:
        return ""
    accelerator = (inspect_torch(torch) if torch_module is not None
                   else resolve())
    calls = {"cuda": "torch.cuda.empty_cache()",
             "rocm": "torch.cuda.empty_cache()",
             "mps": "torch.mps.empty_cache()",
             "xpu": "torch.xpu.empty_cache()"}
    made = calls.get(accelerator.kind, "")
    try:
        if accelerator.kind in ("cuda", "rocm"):
            torch.cuda.empty_cache()
        elif accelerator.kind == "mps":
            torch.mps.empty_cache()
        elif accelerator.kind == "xpu":
            torch.xpu.empty_cache()
    except Exception:                                        # noqa: BLE001
        LOG.debug("empty_cache failed for %s", accelerator.kind,
                  exc_info=True)
        return ""
    return made


def capabilities(found: Optional[Accelerator] = None) -> Tuple[Tuple[str, bool, str], ...]:
    """``(task, accelerated, detail)`` rows stating which tasks are accelerated.

    The setup screen reports acceleration per task rather than a single
    "compatible GPU" flag, because backends differ by task: on Metal,
    segmentation and the classifier are accelerated while the cuML
    reductions run on the CPU.

    Ordered by how much the acceleration is worth: segmentation on the CPU
    took 444 s for one 256x256 image on the machine this was written on,
    and 3.2 s on its Radeon.

    :param found: optional existing accelerator snapshot. Providing it avoids
        another resolution and any dtype-probe allocations, as required by
        the doctor's metadata-only check.
    :returns: task, accelerated flag and explanation for each capability.
    """
    found = resolve() if found is None else found
    gpu = found.is_gpu
    display_gpu = _opengl_likely()
    metal_reducers = gpu and found.kind == 'mps' and not _apple_gpu(found)
    rows = [
        ("Segmentation (Cellpose)", gpu,
         "on the GPU — minutes per image on a CPU" if gpu
         else "CPU only — expect minutes per image"),
        ("Model training", gpu,
         "on the GPU" if gpu else "CPU only — slow but works"),
        ("Model inference / classification", gpu,
         "on the GPU" if gpu else "CPU only"),
        ("Live backdrop and spaceout", display_gpu,
         "GPU shader" if display_gpu else "CPU renderer"),
        ("UMAP / t-SNE / clustering", gpu and found.kind == 'cuda' or metal_reducers,
         "on the GPU via cuML" if gpu and found.is_cuda
         else "PCA/KMeans on Metal; PCA fitting/precision and UMAP/t-SNE/DBSCAN on CPU" if metal_reducers
         else "CPU — cuML is built for CUDA only"),
    ]
    return tuple(rows)


def _apple_gpu(found: Accelerator) -> bool:
    """Whether a Metal accelerator is Apple Silicon's own GPU.

    Read from the resolved name ("Apple M2"), not from the host, so an
    Accelerator describing an Intel Mac's Radeon means that on any machine.
    "Metal GPU" is :func:`_metal_gpu_name`'s fallback when system_profiler
    would not answer; only then does the host decide.
    """
    if found.kind != "mps":
        return False
    name = found.name or ""
    return name.startswith("Apple ") or (name == "Metal GPU" and _apple_silicon())


def _opengl_likely() -> bool:
    """Whether the backdrop's shader path will be taken. Never raises."""
    try:
        from .qt.widgets.fractal_travel import gpu_is_available

        if gpu_is_available():
            return True
        from .qt.widgets.ambient import _flow_graphics_preflight

        return bool(_flow_graphics_preflight())
    except Exception:                                        # noqa: BLE001
        return False


def describe() -> str:
    """One line for a log or a console: what was found and whether it runs."""
    accelerator = resolve()
    emulated = (" — Intel Python under Rosetta on Apple Silicon, see "
                "spacr-doctor" if _rosetta_translated() else "")
    if accelerator.is_gpu:
        return f"{accelerator.label} — in use{emulated}"
    if accelerator.detected and not accelerator.usable:
        return f"{accelerator.label} — found, not used ({accelerator.note}){emulated}"
    return "CPU" + (f" ({accelerator.note})" if accelerator.note else "") + emulated
