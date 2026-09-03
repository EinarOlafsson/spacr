"""One answer to "what should this run on, and what is it called".

spaCR grew up on NVIDIA, so roughly forty-five call sites
ask ``torch.cuda.is_available()`` and most of them mean "is there a GPU?".
Those are DIFFERENT QUESTIONS on every machine that is not NVIDIA, and
conflating them is what makes a perfectly good card read as "no GPU":

* ROCm answers ``torch.cuda.is_available()`` **True** and wants
  ``device="cuda"``. Code that then prints "CUDA" is naming a vendor that
  is not there; code that prints "no NVIDIA GPU" is denying one that is.
* Apple's Metal answers it **False** and wants ``device="mps"``. MPS is
  *Metal*, not an Apple-Silicon feature. On
  the Intel iMac this was written on it drives an AMD Radeon Pro 5300 --
  measured 16x on conv2d inference and 2.7x on a training step against
  the same machine's CPU. ROCm has no macOS build at all, so Metal is the
  ONLY route to an AMD card on a Mac.

So this module answers both questions separately, and a third that matters
more than either: *what is this device unable to do?* Re-pointing a tensor
at MPS is not enough, because three things fail rather than degrade, and
each was measured rather than assumed:

* ``float64`` raises ``TypeError`` outright -- not a slow path, a hard
  stop. Anything wanting double precision must stay on the CPU.
* ``autocast(device_type="mps")`` raises ``RuntimeError``. Mixed precision
  has to be branched off, not merely re-pointed.
* Some operators are simply missing (``aten::linalg_qr.out`` among them).
  ``PYTORCH_ENABLE_MPS_FALLBACK=1`` turns those from a crash into a quiet
  CPU detour, so this module sets it when it selects MPS.

DETECTED AND USABLE ARE REPORTED SEPARATELY. A setup screen announcing an
accelerator spaCR will never dispatch to is worse than one that says nothing,
because the user then blames their hardware for CPU speed. Neural engines are
the whole reason that distinction exists here: the Apple Neural Engine, Intel
AI Boost and Qualcomm Hexagon are all real silicon with no portable torch
device, so they are named as FOUND and never selected.

NOTHING HERE RAISES. A half-installed ROCm, a broken driver, a torch built
without a backend it advertises -- all resolve to the CPU with a note. The
machines most likely to have a strange accelerator are the least able to
afford a traceback at import time.
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
    """What spaCR will compute on, and what it can be told about it.

    :param kind: one of :data:`KINDS`. ``rocm`` is reported distinctly from
        ``cuda`` even though both dispatch to ``device="cuda"``, because
        the only thing a user can act on is the true vendor.
    :param device: the string to hand ``torch.device``.
    :param label: human text for the setup slide and the doctor.
    :param detected: the hardware is present.
    :param usable: spaCR will actually dispatch to it. Never true unless
        ``detected``; false for accelerators with no torch device.
    :param note: why it is not usable, when it is not. Empty otherwise.
    :param float64: the device accepts double precision.
    :param autocast: ``torch.autocast`` accepts this device type.
    :param fallback: missing operators silently run on the CPU instead of
        raising, because this module set the backend's fallback flag.
    :param bfloat16: the device accepts ``torch.bfloat16``. PROBED rather
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
        """Is there a usable non-CPU device.

        THE QUESTION MOST CALL SITES ACTUALLY MEANT when they wrote
        ``torch.cuda.is_available()``.
        """
        return self.usable and self.kind != "cpu"

    @property
    def is_cuda(self) -> bool:
        """Is this genuinely NVIDIA CUDA.

        Narrower than :attr:`is_gpu` and deliberately so: anything reading
        ``nvidia-smi``, a CUDA version, or NVIDIA-specific memory
        interfaces wants this one and would be wrong on ROCm.
        """
        return self.kind == "cuda"

    @property
    def torch_device(self):
        """A ``torch.device`` for :attr:`device`, or the CPU if torch is
        missing entirely."""
        import torch

        return torch.device(self.device)


_CPU = Accelerator(
    kind="cpu", device="cpu", label="CPU", detected=True, usable=True,
    note="", float64=True, autocast=True, fallback=False)

#: Resolved once. Probing costs a CUDA context on some drivers, and the
#: answer cannot change inside a process.
_CACHED: Optional[Accelerator] = None


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


def _cuda_or_rocm(torch) -> Optional[Accelerator]:
    """NVIDIA, or AMD-on-Linux, both of which dispatch to ``cuda``."""
    try:
        if not torch.cuda.is_available():
            return None
    except Exception:                                        # noqa: BLE001
        return None
    # ROCm builds set torch.version.hip and leave torch.version.cuda None.
    # `torch.version` is fetched defensively rather than dotted into: a
    # partially-built torch, and any stand-in that implements only the
    # `cuda` namespace, has no `version` at all -- and an AttributeError
    # here would demote a working CUDA card to the CPU.
    version_module = getattr(torch, "version", None)
    hip = getattr(version_module, "hip", None)
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:                                        # noqa: BLE001
        name = "GPU"
    if hip:
        return Accelerator(
            kind="rocm", device="cuda:0",
            label=f"{name} (AMD ROCm {hip})", name=name,
            # ROCm is a full CUDA-shaped backend: double precision and
            # autocast both work, which is why it needs no capability
            # carve-outs the way Metal does.
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
            # Built but unavailable is a real and confusing state, and it
            # has SEVERAL causes that a user can act on differently. This
            # used to answer all of them with "this system does not offer a
            # Metal device", which on an Intel Mac with Intel graphics is
            # simply false -- the machine has a Metal device, drives its
            # display with it, and torch still cannot use it. Reported by
            # the maintainer on a 2020 Intel Mac.
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
        # MEASURED, not assumed -- see this module's docstring.
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
    intel_mac = platform.machine() in ("x86_64", "i386")

    # ASKED FIRST, because on macOS 12.2 an Apple Silicon Mac reaches here
    # too and "upgrade macOS" is the true answer for it as well.
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


def _xpu(torch) -> Optional[Accelerator]:
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
        name = xpu.get_device_name(0)
    except Exception:                                        # noqa: BLE001
        name = "Intel GPU"
    # Intel's XPU backend has no float64 on most consumer parts and its
    # autocast support depends on the torch build, so both are claimed
    # conservatively: a wrong "yes" here is a crash in a training run.
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
    """Inference silicon that is present and that spaCR will NOT use.

    Reported so the setup slide can say "found, not used" rather than
    leaving a user to wonder why their Neural Engine is idle. There is no
    portable torch device for any of these -- the ANE is reachable only
    through CoreML, Intel's AI Boost only through OpenVINO -- so naming
    one as a compute device would be a promise nothing behind it can keep.
    """
    found = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        found.append("Apple Neural Engine (CoreML only)")
    return tuple(found)


def inspect_torch(torch) -> Accelerator:
    """Resolve against a SPECIFIC torch module, without touching the cache.

    For callers that already hold a torch handle and must be answered about
    that one -- :func:`spacr.doctor.check_gpu` is the case: it imports torch
    through its own indirection so the diagnosis can be exercised against a
    stand-in, and a cached answer about the real machine would defeat that
    entirely. Same probes and same order as :func:`resolve`, so the two
    cannot drift.
    """
    found = None
    for probe in (lambda: _cuda_or_rocm(torch), lambda: _mps(torch),
                  lambda: _xpu(torch), _directml):
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
            # A backend that throws while being ASKED whether it exists is
            # exactly the half-installed case this must survive.
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
        # THE FLAG ITSELF IS SET IN `spacr/__init__.py`, NOT HERE. torch
        # reads it when the MPS backend registers, which is at `import
        # torch` -- long before this resolver runs. Setting it now would
        # look like it worked and change nothing; measured. What is
        # recorded here is only whether the fallback is in force, so a
        # caller can report it.
        found = replace(found, fallback=bool(
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")))
    _CACHED = found
    LOG.debug("accelerator resolved to %s (%s)", found.device, found.label)
    return found


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


# ---------------------------------------------------------------------------
# The shorthands call sites actually want
# ---------------------------------------------------------------------------

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
    """Is there a usable accelerator of any vendor.

    What a call site means when it writes ``torch.cuda.is_available()`` to
    decide whether to use a GPU at all.
    """
    return resolve().is_gpu


def is_cuda() -> bool:
    """Is the resolved accelerator genuinely NVIDIA CUDA.

    For the places that legitimately need CUDA specifically -- memory
    interrogation, ``nvidia-smi`` advice, CUDA version reporting.
    """
    return resolve().is_cuda


def supports_float64() -> bool:
    """Whether double precision may be sent to the device."""
    return resolve().float64


def supports_autocast() -> bool:
    """Whether ``torch.autocast`` accepts this device type."""
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
    """What to pass cellpose as ``gpu=``.

    Cellpose does its own device resolution and already knows about MPS --
    ``assign_device(gpu=True)`` answers ``mps`` on a Metal machine. What it
    cannot do is guess, so it must be TOLD there is a GPU. Passing
    ``torch.cuda.is_available()`` here is what pinned every Mac to the CPU:
    cellpose branches on this flag before it looks at ``device`` at all.
    """
    return is_gpu()


def supports_bfloat16() -> bool:
    """Whether ``torch.bfloat16`` tensors may be put on the device."""
    return resolve().bfloat16


def cellpose_kwargs() -> dict:
    """Everything ``CellposeModel`` needs to land on this machine's GPU.

    THREE ARGUMENTS THAT HAVE TO AGREE, which is why they are produced
    together rather than spelled out at six call sites:

    * ``gpu`` -- cellpose branches on this BEFORE it looks at ``device``,
      so a device without the flag still takes the CPU path.
    * ``device`` -- from the one resolver, so cellpose and spaCR cannot
      disagree about the same machine.
    * ``use_bfloat16`` -- cpsam loads its weights in bfloat16 by default
      and Metal on torch 2.2 has no bfloat16, so the default is a
      ``TypeError: BFloat16 is not supported on MPS`` at construction.
      Measured on the reporting iMac; float32 weights work there and cost
      VRAM, which is the right trade for a card that otherwise sits idle.

    Callers that pass ``device=None`` on purpose -- letting cellpose
    resolve it -- should take ``gpu`` and ``use_bfloat16`` from here and
    drop ``device``.
    """
    accelerator = resolve()
    if accelerator.kind == "mps":
        _keep_cellpose_flows_off_metal()
    kwargs = {"gpu": accelerator.is_gpu, "device": accelerator.torch_device}
    if accelerator.is_gpu and not accelerator.bfloat16:
        kwargs["use_bfloat16"] = False
    return kwargs


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
        return original(masks, flows, threshold=threshold,
                        device=torch.device("cpu"))

    on_the_cpu._spacr_cpu_flows = True
    on_the_cpu.__doc__ = original.__doc__
    dynamics.remove_bad_flow_masks = on_the_cpu
    LOG.debug("cellpose flow-error pass pinned to the CPU (Metal indexing bug)")


def empty_cache(torch_module=None) -> str:
    """Hand the driver back whatever this backend caches. Never raises.

    :param torch_module: ask about THIS torch instead of the cached answer
        for the machine. :mod:`spacr.qt.resource_cleanup` holds its own
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
    # THE CALL IS RETURNED, NOT A GENERIC PHRASE. Preferences shows this
    # verbatim, and "torch.cuda.empty_cache()" is the line a user can look
    # up; "device cache released" is a euphemism they cannot check.
    return made


def capabilities() -> Tuple[Tuple[str, bool, str], ...]:
    """``(task, accelerated, detail)`` for what this machine can actually do.

    WHAT THE SETUP SCREEN IS FOR. "Compatible GPU" on its own answers a
    question nobody asked: users want to know whether THE SLOW STEP will
    be slow. So this reports per task, and it reports the truth per
    backend rather than one verdict for all of them -- on Metal the
    segmentation and the classifier are accelerated while the cuML
    reductions are not, and a single green tick would be a lie about the
    second.

    Ordered by how much the acceleration is worth: segmentation on the CPU
    took 444 s for one 256x256 image on the machine this was written on,
    and 3.2 s on its Radeon.
    """
    found = resolve()
    gpu = found.is_gpu
    rows = [
        ("Segmentation (Cellpose)", gpu,
         "on the GPU — minutes per image on a CPU" if gpu
         else "CPU only — expect minutes per image"),
        ("Model training", gpu,
         "on the GPU" if gpu else "CPU only — slow but works"),
        ("Model inference / classification", gpu,
         "on the GPU" if gpu else "CPU only"),
        ("Live backdrop and spaceout", True,
         "GPU shader" if _opengl_likely() else "CPU renderer"),
        # cuML IS NOT PORTABLE, and saying so here is the point. RAPIDS
        # ships for CUDA only -- there is no AMD, Intel or macOS build --
        # so this row is red on every machine in this list except NVIDIA,
        # and a user on Metal should not wait for it to get faster.
        ("UMAP / t-SNE / clustering", found.is_cuda,
         "on the GPU via cuML" if found.is_cuda
         else "CPU — cuML is built for CUDA only"),
    ]
    return tuple(rows)


def _opengl_likely() -> bool:
    """Whether the backdrop's shader path will be taken. Never raises."""
    try:
        from .qt.widgets.fractal_travel import gpu_is_available

        return bool(gpu_is_available())
    except Exception:                                        # noqa: BLE001
        return False


def describe() -> str:
    """One line for a log or a console: what was found and whether it runs."""
    accelerator = resolve()
    if accelerator.is_gpu:
        return f"{accelerator.label} — in use"
    if accelerator.detected and not accelerator.usable:
        return f"{accelerator.label} — found, not used ({accelerator.note})"
    return "CPU" + (f" ({accelerator.note})" if accelerator.note else "")
