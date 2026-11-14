"""Registering one field across the sequencing cycles that imaged it.

Registration across cycles differs from registration across tiles.
:mod:`spacr.ops_register` places two different fields that overlap by a
fraction of a tile; here the same field is imaged eleven times and the stage
returns to it, so the displacement is a handful of pixels rather than a
pitch.

Cycle registration cannot use Hoechst. Measured on the reference
acquisition: cycle 1 holds one five-channel ``DAPI-CY3-A594-CY5-CY7`` stack
per site, and cycles 2 to 11 hold four separate base-channel files with no
DAPI channel.
So the channel every tile-axis registration runs on does not exist on this
axis after the first cycle, and averaging Hoechst across cycles is not
available here either.

The target used instead comes from the reference implementation.
``ops/firesnake.py:_align_SBS`` offers ``method='SBS_mean'`` for exactly
this case: the maximum across the base channels, normalised by a high
percentile and clipped, which turns four sparse spot channels into one
dense target. Measured on well A1 at two sites, cycles 2 to 11 against
cycle 1:

    site 100   peak/mean 169 to 236   shift (-6..-27,  0..10)
    site 200   peak/mean 218 to 281   shift (-4..-25, 11..21)

    20 of 20 cycles registered.

This target gives stronger correlation peaks than DAPI: tile registration
on DAPI peaks at 23 to 85; this target peaks at 169 to 281. Spots are
punctate and abundant while nuclei are smooth and few, so the missing
Hoechst channel does not reduce registration quality.

The offset depends on the site, not only on the cycle. Site 100 sits at
x = 0 to 10 across the eleven cycles and site 200 at x = 11 to 21. One rigid
transform per cycle for a whole well would carry that difference into every
field, so each site is registered against its own reference -- which is
also what the reference implementation does, one field's cycle stack at a
time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ops_register import Registration, _read_peak, phase_correlate

LOG = logging.getLogger(__name__)

#: The percentile the target is divided by before clipping. The reference
#: uses 70; a spot channel is mostly background, so a high percentile still
#: lands inside the noise and the division brings the spots to order 1.
NORMALISE_PERCENTILE: float = 70.0

#: How far a cycle may move and still be believed, in pixels. The stage
#: returns to the same field, so this is a stage-repeatability number and
#: not a pitch: the measured spread on well A1 is 4 to 27 px, and 64 leaves
#: room for a worse stage without admitting a wrong field.
CYCLE_TOLERANCE: int = 64


def sbs_mean_target(planes: Sequence[np.ndarray], *,
                    percentile: float = NORMALISE_PERCENTILE,
                    cutoff: float = 1.0) -> np.ndarray:
    """Turn one cycle's base channels into a single registration target.

    The maximum over channels is used, not the mean. A read is bright in exactly
    one channel per cycle -- that is what a base call is -- so averaging
    the four divides every spot by four and leaves the background where it
    was. The maximum keeps each spot at its own brightness whichever
    channel carried it.

    :param planes: the base channels for one cycle of one field.
    :param percentile: the percentile the result is divided by.
    :param cutoff: the value the result is clipped at afterwards, so a
        saturated spot cannot dominate the correlation on its own.
    :returns: a float32 image, the same shape as one plane.
    :raises ValueError: when no planes are given, or they disagree in shape.
    """
    stack = [np.asarray(plane, dtype=np.float32) for plane in planes]
    if not stack:
        raise ValueError("sbs_mean_target needs at least one channel")
    if any(plane.shape != stack[0].shape for plane in stack):
        raise ValueError(
            "the base channels of one cycle must share a shape: "
            f"{[plane.shape for plane in stack]}"
        )
    return _normalised_target(np.max(np.stack(stack), axis=0),
                              percentile=percentile, cutoff=cutoff)


def _normalised_target(target: np.ndarray, *,
                       percentile: float = NORMALISE_PERCENTILE,
                       cutoff: float = 1.0) -> np.ndarray:
    """The second half of :func:`sbs_mean_target`, on a channel maximum.

    :param target: the float32 maximum over one cycle's channels.
    :param percentile: the percentile the result is divided by.
    :param cutoff: the value the result is clipped at afterwards.
    :returns: the normalised, clipped float32 target.
    """
    scale = float(np.percentile(target, percentile))
    if scale > 0:
        target = target / scale
    return np.clip(target, 0.0, cutoff).astype(np.float32)


def register_cycles(reference: np.ndarray,
                    cycles: Mapping[int, np.ndarray], *,
                    tolerance: int = CYCLE_TOLERANCE,
                    **kwargs) -> Dict[int, Registration]:
    """Place every cycle of one field against that field's reference.

    Each cycle is registered against the reference directly, never against
    its predecessor. Chaining c1->c2->c3 saves one registration and makes
    every later cycle depend on every earlier one: a single bad link
    displaces the rest and the residual does not say which link failed.
    Registering each cycle independently lets one cycle be dropped without
    affecting the others, so a cycle that does not register does not cost
    the well.

    :param reference: the reference cycle's target, from
        :func:`sbs_mean_target`.
    :param cycles: cycle number to that cycle's target.
    :param tolerance: how far a cycle may land from the reference and still
        be accepted, in pixels.
    :param kwargs: passed to :func:`spacr.ops_register.phase_correlate`.
    :returns: cycle number to its :class:`Registration`, including the
        refused ones -- a caller reporting `n_cycles` needs to know a cycle
        was tried and refused, not merely that it is absent.
    """
    out: Dict[int, Registration] = {}
    for cycle, target in cycles.items():
        out[cycle] = phase_correlate(
            reference, target, expected=(0, 0), tolerance=tolerance, **kwargs)
    return out


def accepted_cycles(registrations: Mapping[int, Registration]) -> Tuple[int, ...]:
    """The cycles whose registration was accepted, in ascending order.

    :param registrations: the result of :func:`register_cycles`.
    :returns: the accepted cycle numbers, ascending.
    """
    return tuple(sorted(c for c, r in registrations.items() if r.accepted))


#: How far one base channel may sit from the first channel of its own cycle
#: and still be moved, in pixels (372 PART 14-L: shifts up to 14 px, 66 of 66
#: accepted at 16). Private: it is a property of this filter set and stage,
#: recorded here rather than offered as a knob nobody has measured.
_CHANNEL_TOLERANCE: int = 16

#: The percentile a channel is clipped at before channel-to-channel
#: registration. Spots differ between channels by definition -- a read is
#: bright in one of them -- so what the four share is the cell background,
#: and clipping the spots away is what lets the correlation see it.
_CHANNEL_CLIP_PERCENTILE: float = 90.0


def _translate(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Move an image's content by whole pixels, filling the gap with zeros.

    :param image: an array whose last two axes are rows and columns.
    :param dy: rows to move down; negative moves up.
    :param dx: columns to move right; negative moves left.
    :returns: the moved copy, the same shape and dtype.

    Integer and zero-filled on purpose: a registration measured to the pixel
    is applied to the pixel, and an interpolated edge would invent intensity
    that a base call then reads.
    """
    dy, dx = int(dy), int(dx)
    out = np.zeros_like(image)
    height, width = image.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        return out
    out[..., max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = \
        image[..., max(-dy, 0):height + min(-dy, 0),
              max(-dx, 0):width + min(-dx, 0)]
    return out


def _clipped(plane: np.ndarray) -> np.ndarray:
    """A channel with its spots clipped off, for channel-to-channel registration.

    :param plane: one base channel.
    :returns: the plane clipped between its minimum and
        :data:`_CHANNEL_CLIP_PERCENTILE`, as float32.
    """
    plane = np.asarray(plane, dtype=np.float32)
    low, high = np.percentile(plane, [0.0, _CHANNEL_CLIP_PERCENTILE])
    return np.clip(plane, low, high)


def _translate_into(out: np.ndarray, image: np.ndarray, dy: int,
                    dx: int) -> np.ndarray:
    """:func:`_translate` written into a preallocated array (item 932).

    :param out: the destination, the shape of ``image``; overwritten.
    :param image: the array to move.
    :param dy: rows to move down; negative moves up.
    :param dx: columns to move right; negative moves left.
    :returns: ``out``, equal to ``_translate(image, dy, dx)``.

    A field moves 44 planes several times over; writing each into one
    buffer rather than a fresh array is a quarter of the field step's time
    on a 1,480 px field, with the same values. :func:`align_field` keeps
    each cycle's :func:`sbs_mean_target` and the spot target as running
    maxima over planes moved into one scratch buffer, and in ``background``
    mode still moves a channel twice -- channel shift, then cycle shift --
    so a row the first move pushed out stays empty after the second brings
    it back, as before.
    """
    dy, dx = int(dy), int(dx)
    out[...] = 0
    height, width = image.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        return out
    out[..., max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = \
        image[..., max(-dy, 0):height + min(-dy, 0),
              max(-dx, 0):width + min(-dx, 0)]
    return out


@dataclass(frozen=True)
class AlignedField:
    """One field's sequencing cycles, every channel moved into one frame.

    :ivar stack: ``(cycles, channels, Y, X)`` float32, one entry per cycle
        in :attr:`kept` and in that order.
    :ivar kept: the cycle numbers the stack holds, ascending. The reference
        cycle is always among them.
    :ivar refused: cycles whose registration against the reference was not
        accepted. They are left out of the stack, because base calls read at
        untrusted coordinates decode to a wrong barcode rather than to none.
    :ivar missing: cycles offered with no planes at all -- a file that could
        not be read, for instance.
    :ivar channel_shifts: ``cycle -> one (dy, dx) per channel``, the shift
        that moved each channel onto the first channel of its cycle.
    :ivar cycle_shifts: ``cycle -> (dy, dx)``, the shift that moved each kept
        cycle onto the reference; the reference's own is ``(0, 0)``.
    :ivar refused_channels: ``(cycle, channel)`` pairs whose within-cycle
        registration was refused and which were therefore not moved.
    :ivar spot_replaced: ``(cycle, channel)`` pairs whose background shift
        the spot registration overruled (``channel_mode='spots'`` only).
    """

    stack: np.ndarray
    kept: Tuple[int, ...]
    refused: Tuple[int, ...]
    missing: Tuple[int, ...]
    channel_shifts: Dict[int, Tuple[Tuple[int, int], ...]]
    cycle_shifts: Dict[int, Tuple[int, int]]
    refused_channels: Tuple[Tuple[int, int], ...] = ()
    spot_replaced: Tuple[Tuple[int, int], ...] = ()


#: How the channels are registered (``align_field(channel_mode=...)``).
#: ``background`` moves each base channel onto its cycle's first channel on
#: the clipped cell background, then each cycle onto the reference.
#: ``spots`` registers every channel of every cycle DIRECTLY onto the
#: reference cycle's spots: both images are Laplacian-of-Gaussian filtered,
#: so the cell background and the illumination -- which every channel of one
#: stage position shares at zero shift -- are gone and only the reads are left.
#: ``spots_refine`` (alpha, item 873) is ``spots`` that also takes a believed
#: spot answer lying within ``refine_px`` of the background answer, so a
#: one- or two-pixel residual is corrected rather than kept; it is the "S2"
#: registration of the decode-yield study.
CHANNEL_MODES = ("background", "spots", "spots_refine")

#: How close a believed spot answer must lie to the background answer to
#: refine it under ``spots_refine``, in pixels.
_SPOT_REFINE_PX = 3

#: How far a channel's spot registration may disagree with its background
#: registration before the spot answer replaces it, in pixels. Within this the
#: two agree and the background answer stands, so a good field is unchanged.
_SPOT_AGREEMENT_PX = 2

#: The weakest spot-registration peak believed, against the surface mean.
#: Measured on the Funk et al. 2022 plates (2026-11-04): a true spot peak on a
#: 1,480 px field stands at hundreds; a field with no reads at a few.
_SPOT_MIN_PEAK = 20.0

#: The spots overrule the background only when the spot surface at the
#: background's shift is below this share of the spot peak. Where the
#: background is right the reads coincide there too and the surface is near
#: its peak; where it locked onto the zero-shift pattern the reads do not
#: coincide and the surface there is noise.
_SPOT_OVERRULE_SHARE = 0.5


#: The precisions the registration transforms may run in (item 932).
#: ``float32`` is the default: on 443 Funk et al. 2022 fields it gave the
#: same integer shifts, reads and calls as ``float64`` at about half the
#: time. Identical there, not by construction, so ``float64`` -- the
#: arithmetic before item 932, bit for bit -- stays available.
FFT_PRECISIONS = ("float32", "float64")

#: Where the registration transforms may run (item 932). ``gpu`` batches
#: each cycle's transforms through ``torch.fft`` on a CUDA card and falls
#: back to the CPU when there is none.
FFT_BACKENDS = ("cpu", "gpu")

#: The default precision and backend. On the S2 speed card's ten Funk et al.
#: 2022 fields (item 932, one core, same run) the GPU float32 field step took
#: 8.0 CPU s + 0.6 GPU s per field against 12.0 CPU s on the CPU, faster on
#: CPU and wall time on 9 of 10 fields, so the GPU is the default wherever
#: one is usable and the CPU everywhere else.
FFT_PRECISION = "float32"
FFT_BACKEND = "gpu"


def _cuda_torch(precision: str):
    """``(torch, device)`` for the GPU backend, or None to use the CPU.

    :param precision: the precision asked for; a card without float64
        (Metal) cannot run ``float64``.
    :returns: torch and its CUDA device when spaCR resolves a usable CUDA
        (or ROCm) card, else None.
    """
    try:
        import torch

        from .accelerator import resolve

        found = resolve()
        if not found.is_gpu or not str(found.device).startswith("cuda") \
                or not torch.cuda.is_available():
            return None
        if precision == "float64" and not found.float64:
            return None
        return torch, torch.device(found.device)
    except Exception:                                    # noqa: BLE001
        return None


def _fft_backend_used(backend: str = FFT_BACKEND, gpu: bool = True,
                     precision: str = FFT_PRECISION) -> str:
    """Where the registration transforms of :func:`align_field` will run.

    :param backend: one of :data:`FFT_BACKENDS`.
    :param gpu: False keeps them on the CPU whatever ``backend`` says.
    :param precision: one of :data:`FFT_PRECISIONS`.
    :returns: ``"gpu"`` when ``backend`` is ``"gpu"``, ``gpu`` is True and
        a CUDA card is usable through torch; ``"cpu"`` otherwise, which is
        the automatic fallback.
    :raises ValueError: for an unknown backend or precision.
    """
    if backend not in FFT_BACKENDS:
        raise ValueError(f"backend must be one of {list(FFT_BACKENDS)}; "
                         f"got {backend!r}")
    if precision not in FFT_PRECISIONS:
        raise ValueError(f"precision must be one of {list(FFT_PRECISIONS)}; "
                         f"got {precision!r}")
    if backend == "gpu" and gpu and _cuda_torch(precision) is not None:
        return "gpu"
    return "cpu"


class _FFTEngine:
    """The registration transforms of one field, on the CPU or a CUDA card.

    Every surface is the one :func:`spacr.ops_register.phase_correlate`
    (background) or the spot registration computes, and each reference's
    forward transform is computed once and reused for every partner: the
    first channel of a cycle for its three others, the reference cycle's
    target for the ten cycles, the spot target for the 44 spot planes.

    CPU ``float64`` is ``numpy.fft``, so its surfaces are bit for bit those
    before item 932; CPU ``float32`` is ``scipy.fft`` on one thread. The GPU
    path batches one cycle's transforms through ``torch.fft`` in the same
    precision, with deterministic algorithms asked for while it runs, and
    drops to the CPU for the rest of the field if the card fails.

    :param precision: one of :data:`FFT_PRECISIONS`.
    :param backend: one of :data:`FFT_BACKENDS`.
    :param gpu: False keeps it on the CPU.
    """

    def __init__(self, precision: str = FFT_PRECISION,
                 backend: str = FFT_BACKEND, gpu: bool = True):
        """Resolve where the transforms run, once for the field.

        :param precision: one of :data:`FFT_PRECISIONS`.
        :param backend: one of :data:`FFT_BACKENDS`.
        :param gpu: False keeps it on the CPU.
        """
        self.precision = precision
        self.used = _fft_backend_used(backend, gpu, precision)
        self.dtype = np.float32 if precision == "float32" else np.float64
        self.torch = self.device = None
        if self.used == "gpu":
            self.torch, self.device = _cuda_torch(precision)

    def _forward(self, image: np.ndarray):
        """The real 2-D forward transform on the CPU, in the precision.

        :param image: a 2-D image.
        :returns: its half spectrum.
        """
        image = np.asarray(image, dtype=self.dtype)
        if self.dtype == np.float64:
            return np.fft.rfft2(image)
        import scipy.fft

        return scipy.fft.rfft2(image, workers=1)

    def _inverse(self, spectrum, shape):
        """The real 2-D inverse transform on the CPU.

        :param spectrum: a half spectrum.
        :param shape: the image shape to return.
        :returns: the real image.
        """
        if self.dtype == np.float64:
            return np.fft.irfft2(spectrum, s=shape)
        import scipy.fft

        return scipy.fft.irfft2(spectrum, s=shape, workers=1)

    def _cpu_fallback(self, error: Exception) -> None:
        """Move the rest of the field to the CPU after the card failed.

        :param error: what the GPU path raised, logged.
        """
        LOG.warning("the GPU registration failed (%s); this field continues "
                    "on the CPU", error)
        self.used, self.torch, self.device = "cpu", None, None

    def _gpu(self, function, *args):
        """Run a batched GPU function with deterministic algorithms on."""
        torch = self.torch
        before = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            return function(*args)
        finally:
            torch.use_deterministic_algorithms(before, warn_only=warn)

    def _tensor(self, images):
        """Stack images into one tensor on the card, in the precision.

        :param images: same-shape 2-D images.
        :returns: an ``(N, H, W)`` tensor.
        """
        torch = self.torch
        stacked = np.stack([np.asarray(image) for image in images])
        dtype = torch.float32 if self.dtype == np.float32 else torch.float64
        return torch.as_tensor(np.ascontiguousarray(stacked),
                               device=self.device).to(dtype)

    def register(self, reference: np.ndarray, others: Sequence[np.ndarray], *,
                 tolerance, record: Optional[list] = None) -> List[Registration]:
        """``phase_correlate(reference, o, expected=(0, 0), tolerance)`` per o.

        :param reference: the image every other is placed against.
        :param others: the images to place.
        :param tolerance: the largest accepted shift, in pixels.
        :param record: a list each result is appended to as ``(dy, dx,
            peak_ratio, accepted)``, or None.
        :returns: one :class:`spacr.ops_register.Registration` per image.
        """
        others = list(others)
        if not others:
            return []
        peaks = None
        if self.used == "gpu":
            try:
                peaks = self._gpu(self._register_gpu, reference, others)
            except Exception as error:                   # noqa: BLE001
                self._cpu_fallback(error)
        if peaks is None:
            shape = np.shape(reference)
            first = self._forward(reference)
            peaks = []
            for other in others:
                product = first * np.conj(self._forward(other))
                magnitude = np.abs(product)
                with np.errstate(invalid="ignore", divide="ignore"):
                    product = np.where(magnitude > 0, product / magnitude, 0)
                product[0, 0] = 0
                surface = self._inverse(product, shape)
                flat = int(np.argmax(surface))
                peak_y, peak_x = np.unravel_index(flat, surface.shape)
                peaks.append((int(peak_y), int(peak_x),
                              float(surface.flat[flat]),
                              float(np.abs(surface).mean())))
        name = ("torch" if self.used == "gpu" else
                "numpy" if self.dtype == np.float64 else "scipy")
        found = [_read_peak(py, px, height, mean, np.shape(reference),
                            expected=(0, 0), tolerance=tolerance, backend=name)
                 for py, px, height, mean in peaks]
        if record is not None:
            record.extend((int(r.dy), int(r.dx), float(r.peak_ratio),
                           bool(r.accepted)) for r in found)
        return found

    def _register_gpu(self, reference, others):
        """:meth:`register`'s surfaces as one batch on the card.

        :param reference: the image every other is placed against.
        :param others: the images to place.
        :returns: per image ``(peak row, peak column, peak, mean)``.
        """
        torch = self.torch
        images = self._tensor([reference] + others)
        spectra = torch.fft.rfft2(images)
        product = spectra[:1] * torch.conj(spectra[1:])
        magnitude = torch.abs(product)
        floor = 1e-300 if self.dtype == np.float64 else 1e-30
        product = torch.where(magnitude > 0,
                              product / torch.clamp(magnitude, min=floor),
                              torch.zeros_like(product))
        product[:, 0, 0] = 0
        surfaces = torch.fft.irfft2(product, s=tuple(images.shape[-2:]))
        count, _, width = surfaces.shape
        flat_all = surfaces.reshape(count, -1)
        flat = torch.argmax(flat_all, dim=1)
        height = flat_all.gather(1, flat[:, None])[:, 0]
        mean = surfaces.abs().reshape(count, -1).mean(1)
        flat, height, mean = (v.cpu().numpy() for v in (flat, height, mean))
        return [(int(flat[i]) // width, int(flat[i]) % width,
                 float(height[i]), float(mean[i])) for i in range(count)]

    def spot_peaks(self, target: np.ndarray, spots: Sequence[np.ndarray],
                   near: Sequence[Tuple[int, int]]):
        """Each spot image's phase-correlation peak against the spot target.

        :param target: the reference cycle's spot target.
        :param spots: the spot images to place.
        :param near: one ``(dy, dx)`` per spot image, the background answer.
        :returns: per spot image ``(dy, dx, peak, mean, near)`` -- the
            shift, the peak, the surface's mean absolute value, and the
            surface's largest value within one pixel of the background
            answer (:func:`_surface_near`).
        """
        spots = list(spots)
        if self.used == "gpu":
            try:
                return self._gpu(self._spot_gpu, target, spots, near)
            except Exception as error:                   # noqa: BLE001
                self._cpu_fallback(error)
        if getattr(self, "_target_source", None) is not target:
            self._target, self._target_source = self._forward(target), target
        shape = np.shape(target)
        out = []
        for image, mine in zip(spots, near):
            cross = self._target * np.conj(self._forward(image))
            cross /= np.abs(cross) + 1e-12
            surface = self._inverse(cross, shape)
            flat = int(np.argmax(surface))
            peak = float(surface.flat[flat])
            mean = float(np.abs(surface).mean())
            py, px = np.unravel_index(flat, surface.shape)
            dy = int(py) - surface.shape[0] if py > surface.shape[0] // 2 else int(py)
            dx = int(px) - surface.shape[1] if px > surface.shape[1] // 2 else int(px)
            out.append((dy, dx, peak, mean, _surface_near(surface, mine)))
        return out

    def _spot_gpu(self, target, spots, near):
        """:meth:`spot_peaks` as one batch on the card.

        :param target: the spot target; its transform is kept for reuse.
        :param spots: the spot images.
        :param near: one background answer per spot image.
        :returns: as :meth:`spot_peaks`.
        """
        torch = self.torch
        if getattr(self, "_gpu_target_source", None) is not target:
            self._gpu_target = torch.fft.rfft2(self._tensor([target]))[0]
            self._gpu_target_source = target
        images = self._tensor(spots)
        height, width = images.shape[-2:]
        cross = self._gpu_target[None] * torch.conj(torch.fft.rfft2(images))
        cross = cross / (torch.abs(cross) + 1e-12)
        surfaces = torch.fft.irfft2(cross, s=(height, width))
        count = surfaces.shape[0]
        flat_all = surfaces.reshape(count, -1)
        flat = torch.argmax(flat_all, dim=1)
        peak = flat_all.gather(1, flat[:, None])[:, 0]
        mean = surfaces.abs().reshape(count, -1).mean(1)
        index = torch.as_tensor(
            [[((m[0] + dy) % height) * width + (m[1] + dx) % width
              for dy in (-1, 0, 1) for dx in (-1, 0, 1)] for m in near],
            device=self.device)
        best_near = flat_all.gather(1, index).max(1).values
        flat, peak, mean, best_near = (v.cpu().numpy() for v in
                                       (flat, peak, mean, best_near))
        out = []
        for i in range(count):
            py, px = divmod(int(flat[i]), width)
            dy = py - height if py > height // 2 else py
            dx = px - width if px > width // 2 else px
            out.append((dy, dx, float(peak[i]), float(mean[i]),
                        float(best_near[i])))
        return out


#: One CPU :class:`_FFTEngine` per precision, built on first use.
_CPU_ENGINES: Dict[str, "_FFTEngine"] = {}


def _cpu_engine(precision: str) -> "_FFTEngine":
    """The CPU transform engine for ``precision``, shared by every caller.

    :func:`spacr.ops_register.phase_correlate` computes its float32 surface
    through this engine (item 936), so the stitch and the cycle
    registration use one set of float32 transforms.

    :param precision: one of :data:`FFT_PRECISIONS`.
    :returns: an engine that never uses the card.
    """
    engine = _CPU_ENGINES.get(precision)
    if engine is None:
        engine = _CPU_ENGINES[precision] = _FFTEngine(precision, "cpu",
                                                      gpu=False)
    return engine


def _surface_near(surface: np.ndarray, shift) -> float:
    """The surface's largest value within one pixel of ``shift``."""
    height, width = surface.shape
    return float(max(surface[(shift[0] + dy) % height, (shift[1] + dx) % width]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)))


def _spot_image(plane: np.ndarray) -> np.ndarray:
    """A plane's reads with its background removed: -LoG(1 px), clipped at 0.

    :param plane: one 2-D channel.
    :returns: float32, high only at spot centres.
    """
    from scipy import ndimage

    return np.clip(-ndimage.gaussian_laplace(
        np.asarray(plane, dtype=np.float32), 1.0), 0, None)


def _spot_shifts(planes32, kept, background_total, reference, *, tolerance,
                 engine, refine_px=None, min_peak=_SPOT_MIN_PEAK,
                 overrule_share=_SPOT_OVERRULE_SHARE):
    """Each channel's total shift onto the reference cycle, read off the spots.

    :param planes32: ``cycle -> float32 planes``.
    :param kept: the cycles whose background registration was accepted.
    :param background_total: ``cycle -> [(dy, dx) per channel]``, the
        background answer: cycle shift plus channel shift.
    :param reference: the reference cycle.
    :param tolerance: the largest total shift accepted, in pixels.
    :param engine: the field's :class:`_FFTEngine`.
    :param refine_px: None for the ``spots`` rule; a pixel count for
        ``spots_refine``, under which a believed spot answer within this
        distance of the background answer replaces it as well.
    :param min_peak: the weakest spot peak believed, against the surface
        mean.
    :param overrule_share: the share of the spot peak the surface at the
        background answer must stay below for the spots to overrule it.
    :returns: ``(totals, replaced)`` -- the shift each channel is moved by,
        and the ``(cycle, channel)`` pairs whose background answer the spots
        overruled or refined.

    WHY: on four of the Funk plates' weak wells one cycle's channels came
    back from the background registration at (0, 0) -- the peak every
    channel of one stage position shares through the illumination and the
    camera -- while the true drift was 10 to 25 px, and that one cycle then
    carried 70 to 84 % of the wells' base-call errors. A read is bright in
    exactly one channel of a cycle and in some channel of the reference, so a
    channel's spots are a subset of the reference's and register onto them
    directly, with nothing at zero shift to lock onto.
    """
    ref_spots = [_spot_image(plane) for plane in planes32[reference]]
    target = np.empty_like(ref_spots[0])
    scratch = np.empty_like(ref_spots[0])
    for k, image in enumerate(ref_spots):
        if k == 0:
            _translate_into(target, image, *background_total[reference][k])
        else:
            np.maximum(target, _translate_into(
                scratch, image, *background_total[reference][k]), out=target)
    totals = {}
    replaced = []
    for cycle in kept:
        totals[cycle] = []
        mines = [background_total[cycle][k] for k in range(len(planes32[cycle]))]
        spots = (ref_spots if cycle == reference
                 else [_spot_image(plane) for plane in planes32[cycle]])
        for k, (dy, dx, peak, mean, near) in enumerate(
                engine.spot_peaks(target, spots, mines)):
            mine = mines[k]
            strong = (mean > 0 and peak / mean >= min_peak
                      and abs(dy) <= tolerance and abs(dx) <= tolerance)
            if refine_px is None:
                agree = (abs(dy - mine[0]) <= _SPOT_AGREEMENT_PX
                         and abs(dx - mine[1]) <= _SPOT_AGREEMENT_PX)
                take = strong and not agree and near < overrule_share * peak
            else:
                close = (abs(dy - mine[0]) <= refine_px
                         and abs(dx - mine[1]) <= refine_px)
                overrule = (strong and not close
                            and near < overrule_share * peak)
                take = (strong and (close or overrule)
                        and (dy, dx) != tuple(mine))
            if take:
                totals[cycle].append((dy, dx))
                replaced.append((int(cycle), int(k)))
            else:
                totals[cycle].append(tuple(int(v) for v in mine))
    return totals, replaced


def align_field(planes: Mapping[int, Optional[Sequence[np.ndarray]]], *,
                reference: Optional[int] = None,
                tolerance: int = CYCLE_TOLERANCE,
                gpu: bool = True,
                channel_mode: str = "background",
                refine_px: int = _SPOT_REFINE_PX,
                spot_min_peak: float = _SPOT_MIN_PEAK,
                spot_overrule_share: float = _SPOT_OVERRULE_SHARE,
                precision: str = FFT_PRECISION,
                backend: str = FFT_BACKEND,
                registrations: Optional[list] = None
                ) -> AlignedField:
    """Register one field's channels within each cycle, then its cycles.

    Each base channel is first moved onto its cycle's first channel, on the
    clipped cell background the channels share -- on the measured plate they
    sat up to 14 px apart, which no cycle-to-cycle registration can see --
    and each cycle is then moved onto the reference with
    :func:`sbs_mean_target`. A cycle offered as ``None`` or with a missing
    plane is reported in :attr:`AlignedField.missing` and one whose
    registration is refused in :attr:`AlignedField.refused`; the rest are
    still aligned, so one lost cycle does not cost the field.

    :param planes: ``cycle -> the base channels of that cycle``, every plane
        one 2-D image of the same shape.
    :param reference: the cycle everything is moved onto. None takes the
        lowest cycle that has planes.
    :param tolerance: how far a cycle may land from the reference and still
        be accepted, in pixels.
    :param gpu: allow the ``gpu`` backend to use the card; False keeps every
        transform on the CPU.
    :param channel_mode: one of :data:`CHANNEL_MODES`. ``spots`` runs the
        background registration first and then checks every channel against
        the reference cycle's spots, replacing the shifts the spots disagree
        with; :attr:`AlignedField.spot_replaced` lists those.
        ``spots_refine`` also replaces a believed spot answer within
        ``refine_px`` of the background answer.
    :param refine_px: the refinement reach of ``spots_refine``, in pixels.
    :param spot_min_peak: the weakest spot peak believed, against the
        surface mean (``spots`` and ``spots_refine``).
    :param spot_overrule_share: the share of the spot peak the surface at
        the background answer must stay below for the spots to overrule it.
    :param precision: the registration transforms' precision, one of
        :data:`FFT_PRECISIONS`. ``float64`` is the arithmetic before item 932.
    :param backend: where they run, one of :data:`FFT_BACKENDS`; ``gpu``
        falls back to the CPU when no CUDA card is usable.
    :param registrations: a list each background registration is appended
        to, as ``(dy, dx, peak_ratio, accepted)`` -- the channels of every
        cycle, then the cycles -- for the field diagnostics; None records
        nothing.
    :returns: the aligned field.
    :raises ValueError: when no cycle has planes, when the reference
        cycle has none, or for an unknown ``channel_mode``, ``precision``
        or ``backend``.
    """
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"channel_mode must be one of {list(CHANNEL_MODES)}; "
                         f"got {channel_mode!r}")
    engine = _FFTEngine(precision, backend, gpu=gpu)
    usable = {}
    missing = []
    for cycle in sorted(planes):
        offered = planes[cycle]
        if offered is None or len(offered) == 0 or any(
                plane is None for plane in offered):
            missing.append(int(cycle))
        else:
            usable[int(cycle)] = offered
    if not usable:
        raise ValueError("no cycle of this field has planes to align")
    if reference is None:
        reference = min(usable)
    if reference not in usable:
        raise ValueError(
            f"the reference cycle {reference} has no planes; every other "
            "cycle is placed against it, so choose a cycle that was read")

    planes32 = {cycle: [np.asarray(plane, dtype=np.float32) for plane in offered]
                for cycle, offered in usable.items()}
    channel_shifts = {}
    refused_channels = []
    targets = {}
    for cycle, cycle_planes in planes32.items():
        found = engine.register(
            _clipped(cycle_planes[0]),
            [_clipped(plane) for plane in cycle_planes[1:]],
            tolerance=_CHANNEL_TOLERANCE, record=registrations)
        shifts = [(0, 0)] + [(int(r.dy), int(r.dx)) if r.accepted else (0, 0)
                             for r in found]
        channel_shifts[cycle] = tuple(shifts)
        refused_channels.extend((cycle, index) for index, r in
                                enumerate(found, start=1) if not r.accepted)
        target = cycle_planes[0].copy()
        scratch = np.empty_like(target)
        for plane, shift in zip(cycle_planes[1:], shifts[1:]):
            np.maximum(target, _translate_into(scratch, plane, *shift),
                       out=target)
        targets[cycle] = _normalised_target(target)

    others = [cycle for cycle in usable if cycle != reference]
    registrations_found = dict(zip(others, engine.register(
        targets[reference], [targets[cycle] for cycle in others],
        tolerance=tolerance, record=registrations)))
    del targets
    kept = sorted([reference] + list(accepted_cycles(registrations_found)))
    cycle_shifts = {reference: (0, 0)}
    for cycle in kept:
        if cycle != reference:
            found = registrations_found[cycle]
            cycle_shifts[cycle] = (int(found.dy), int(found.dx))
    replaced = []
    spots = channel_mode in ("spots", "spots_refine")
    if spots:
        background_total = {
            cycle: [(cycle_shifts[cycle][0] + dy, cycle_shifts[cycle][1] + dx)
                    for dy, dx in channel_shifts[cycle]] for cycle in kept}
        totals, replaced = _spot_shifts(
            planes32, kept, background_total, reference,
            tolerance=tolerance + _CHANNEL_TOLERANCE, engine=engine,
            refine_px=(int(refine_px) if channel_mode == "spots_refine"
                       else None),
            min_peak=float(spot_min_peak),
            overrule_share=float(spot_overrule_share))
        for cycle in kept:
            channel_shifts[cycle] = tuple(
                (dy - cycle_shifts[cycle][0], dx - cycle_shifts[cycle][1])
                for dy, dx in totals[cycle])
    first = planes32[reference][0]
    stack = np.empty((len(kept), len(planes32[reference])) + first.shape,
                     np.float32)
    scratch = np.empty_like(first)
    for index, cycle in enumerate(kept):
        for channel, plane in enumerate(planes32[cycle]):
            if spots:
                _translate_into(stack[index, channel], plane,
                                *totals[cycle][channel])
            elif cycle == reference:
                _translate_into(stack[index, channel], plane,
                                *channel_shifts[cycle][channel])
            else:
                _translate_into(stack[index, channel], _translate_into(
                    scratch, plane, *channel_shifts[cycle][channel]),
                    *cycle_shifts[cycle])
    refused = tuple(sorted(c for c, r in registrations_found.items()
                           if not r.accepted))
    return AlignedField(
        stack=stack,
        kept=tuple(kept), refused=refused, missing=tuple(missing),
        channel_shifts=channel_shifts, cycle_shifts=cycle_shifts,
        refused_channels=tuple(refused_channels),
        spot_replaced=tuple(replaced))
