"""The three array operations OPS uses, run on a GPU when one is available.

The GPU-suitable steps of OPS are the per-window affine warp, peak finding, the
unmixing matrix multiply and Cellpose segmentation (which has its own GPU
path), and, for the merge, ORB detection with Hamming descriptor matching and
mutual-nearest-neighbour cell matching. These reduce to three primitives: a
windowed maximum, a matrix multiply and a nearest-neighbour search. They are
collected here so device handling lives in one place.

    Backend order: PyTorch first, CuPy second, NumPy always. Torch is already a
    spaCR dependency through Cellpose, so nothing new is installed, and one
    code path reaches CUDA, ROCm and Apple MPS.

Every function takes and returns NumPy arrays, so callers do not depend on
which backend ran. Each has a test that runs both paths on the same input and
asserts they agree.

The GPU may be shared, so `ops_gpu=False` refuses a device that exists, and a
backend that raises at runtime (an out-of-memory, a driver that went away)
falls through to the next rather than failing the plate.

    The CPU path is a supported path, not an error path: it runs on laptops, in
    CI and when the GPU is busy.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "accelerated_backends",
    "matmul",
    "maximum_filter",
    "nearest_neighbours",
]

#: How many rows of a distance matrix to hold at once. A well can carry
#: 100,000 nuclei, and 100,000 x 100,000 floats is 40 GB -- more than the
#: card and more than the host. The search is chunked over the first set,
#: which bounds the working set at ``CHUNK x M`` whatever the well holds.
CHUNK: int = 4096


def _torch(gpu: bool = True):
    """``(torch, device)``, GPU when spaCR resolves one and ``gpu`` allows.

    Falls back to the CPU device rather than to None: `torch.fft` and
    `torch.cdist` on the CPU are the same code, and refusing them where
    there is no card would leave the accelerated path untested exactly
    where it is most likely to be run.
    """
    try:
        import torch
    except Exception:                                    # noqa: BLE001
        return None
    if not gpu:
        return torch, torch.device("cpu")
    try:
        from .accelerator import is_gpu, torch_device

        if is_gpu():
            return torch, torch_device()
    except Exception:                                    # noqa: BLE001
        LOG.debug("no accelerator resolved for torch", exc_info=True)
    return torch, torch.device("cpu")


def _cupy(gpu: bool = True):
    """CuPy with a working device, or None."""
    if not gpu:
        return None
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return cupy
    except Exception:                                    # noqa: BLE001
        return None


def accelerated_backends(gpu: bool = True) -> Tuple[str, ...]:
    """Which backends can run here, best first, always ending in "numpy".

    :param gpu: False refuses the accelerated ones outright.
    :returns: the usable names.
    """
    found = []
    if gpu:
        found_torch = _torch(True)
        if found_torch is not None and found_torch[1].type != "cpu":
            found.append("torch")
        if _cupy(True) is not None:
            found.append("cupy")
    found.append("numpy")
    return tuple(found)


def _try(order, run, name_of_forced: Optional[str] = None):
    """Run the first backend that does not raise, and say which did.

    :param order: backend names to try, in order.
    :param run: ``name -> result``, raising to fall through.
    :param name_of_forced: when the caller named a backend, its failure is
        raised rather than swallowed -- falling through silently would
        answer a question nobody asked.
    :returns: ``(result, backend name)``.
    """
    last = None
    for name in order:
        try:
            return run(name), name
        except Exception as error:                       # noqa: BLE001
            if name_of_forced:
                raise
            LOG.debug("the %s backend declined this call", name, exc_info=True)
            last = error
    raise RuntimeError(f"no usable backend: {last}")



def maximum_filter(field: np.ndarray, size: int, *, gpu: bool = True,
                   backend: Optional[str] = None) -> np.ndarray:
    """The maximum of each ``size x size`` window, centred on every pixel.

    Used for peak finding: a read is a local maximum of the spot score, and
    "local" means this. On a well-sized field it is also the most expensive
    thing in the decode chain, and it is a max-pool, which GPUs run
    efficiently.

    Edges repeat the border pixel, matching `scipy.ndimage.maximum_filter`
    with ``mode="nearest"``, because the alternative is a rim of false
    maxima all the way round the field.

    :param field: a 2-D array.
    :param size: the window edge, in pixels. Even sizes are raised to the
        next odd one: a window with no centre cannot be centred.
    :param gpu: False keeps it on the CPU.
    :param backend: force one.
    :returns: the filtered field, same shape and dtype-compatible.
    """
    data = np.asarray(field, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"a windowed maximum needs a 2-D field, got "
                         f"{data.shape}")
    window = int(size)
    if window <= 1:
        return data.copy()
    if window % 2 == 0:
        window += 1

    def run(name):
        """The windowed maximum on one named backend.

        :param name: "torch", "cupy" or "numpy".
        :returns: the filtered field as a 2-D array.
        :raises RuntimeError: when that backend is not installed, which
            is what `_try` reads to fall through to the next one.
        """
        if name == "torch":
            found = _torch(gpu)
            if found is None:
                raise RuntimeError("torch is not installed")
            torch, device = found
            tensor = torch.as_tensor(data, device=device)[None, None]
            pad = window // 2
            padded = torch.nn.functional.pad(tensor, (pad, pad, pad, pad),
                                             mode="replicate")
            pooled = torch.nn.functional.max_pool2d(padded, window, stride=1)
            return pooled[0, 0].detach().to("cpu").numpy()
        if name == "cupy":
            cupy = _cupy(gpu)
            if cupy is None:
                raise RuntimeError("cupy is not usable here")
            from cupyx.scipy.ndimage import maximum_filter as cupy_filter

            return cupy.asnumpy(cupy_filter(cupy.asarray(data), size=window,
                                            mode="nearest"))
        from scipy.ndimage import maximum_filter as scipy_filter

        return scipy_filter(data, size=window, mode="nearest")

    order = (backend,) if backend else accelerated_backends(gpu)
    result, _name = _try(order, run, backend)
    return result



def matmul(first: np.ndarray, second: np.ndarray, *, gpu: bool = True,
           backend: Optional[str] = None) -> np.ndarray:
    """``first @ second``, on the first available backend.

    Used for unmixing: correcting dye bleed-through applies one small matrix to
    every spot of every cycle, a large multiply by a tiny operand that suits a
    GPU.

    :param first: left operand.
    :param second: right operand.
    :param gpu: False keeps it on the CPU.
    :param backend: force one.
    :returns: the product, as float32 NumPy.
    """
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)

    def run(name):
        """The matrix product on one named backend.

        :param name: "torch", "cupy" or "numpy".
        :returns: the product as float32 NumPy, wherever it was computed.
        :raises RuntimeError: when that backend is not installed.
        """
        if name == "torch":
            found = _torch(gpu)
            if found is None:
                raise RuntimeError("torch is not installed")
            torch, device = found
            product = (torch.as_tensor(left, device=device)
                       @ torch.as_tensor(right, device=device))
            return product.detach().to("cpu").numpy()
        if name == "cupy":
            cupy = _cupy(gpu)
            if cupy is None:
                raise RuntimeError("cupy is not usable here")
            return cupy.asnumpy(cupy.asarray(left) @ cupy.asarray(right))
        return left @ right

    order = (backend,) if backend else accelerated_backends(gpu)
    result, _name = _try(order, run, backend)
    return np.asarray(result, dtype=np.float32)



def nearest_neighbours(source: np.ndarray, target: np.ndarray, *,
                       gpu: bool = True, backend: Optional[str] = None,
                       chunk: int = CHUNK) -> Tuple[np.ndarray, np.ndarray]:
    """For each row of ``source``, the closest row of ``target``.

    Always chunked. A well can carry a hundred thousand nuclei, and the full
    distance matrix between two such sets is 40 GB -- more than the card and
    more than the host. Chunking the first set bounds the working set at
    ``chunk x M`` however large the well is.

    Ties go to the lower index, on every backend, so a caller that pairs
    mutual nearest neighbours gets the same pairs whatever ran.

    :param source: ``(N, D)`` points.
    :param target: ``(M, D)`` points.
    :param gpu: False keeps it on the CPU.
    :param backend: force one.
    :param chunk: rows of ``source`` per pass.
    :returns: ``(indices, distances)``, both length N.
    """
    src = np.asarray(source, dtype=np.float32)
    dst = np.asarray(target, dtype=np.float32)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != dst.shape[1]:
        raise ValueError(
            f"two sets of points of the same width are needed, got "
            f"{src.shape} and {dst.shape}")
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return (np.empty(src.shape[0], dtype=np.int64),
                np.empty(src.shape[0], dtype=np.float32))

    def run(name):
        """Nearest neighbours on one named backend.

        :param name: "torch", "cupy" or "numpy".
        :returns: ``(indices, distances)``, one row per source point.
        :raises RuntimeError: when that backend is not installed.

        Chunked at :data:`CHUNK` on every backend, because the pairwise
        distance matrix is the memory cost here and it is quadratic.
        """
        indices = np.empty(src.shape[0], dtype=np.int64)
        distances = np.empty(src.shape[0], dtype=np.float32)
        if name == "torch":
            found = _torch(gpu)
            if found is None:
                raise RuntimeError("torch is not installed")
            torch, device = found
            target_tensor = torch.as_tensor(dst, device=device)
            for start in range(0, src.shape[0], chunk):
                block = torch.as_tensor(src[start:start + chunk],
                                        device=device)
                gaps = torch.cdist(
                    block, target_tensor,
                    compute_mode="donot_use_mm_for_euclid_dist")
                best = torch.argmin(gaps, dim=1)
                indices[start:start + chunk] = best.to("cpu").numpy()
                distances[start:start + chunk] = torch.gather(
                    gaps, 1, best[:, None])[:, 0].to("cpu").numpy()
            return indices, distances
        if name == "cupy":
            cupy = _cupy(gpu)
            if cupy is None:
                raise RuntimeError("cupy is not usable here")
            target_array = cupy.asarray(dst)
            for start in range(0, src.shape[0], chunk):
                block = cupy.asarray(src[start:start + chunk])
                gaps = cupy.linalg.norm(block[:, None, :] - target_array[None],
                                        axis=2)
                best = cupy.argmin(gaps, axis=1)
                indices[start:start + chunk] = cupy.asnumpy(best)
                distances[start:start + chunk] = cupy.asnumpy(
                    cupy.take_along_axis(gaps, best[:, None], axis=1)[:, 0])
            return indices, distances
        for start in range(0, src.shape[0], chunk):
            block = src[start:start + chunk]
            gaps = np.linalg.norm(block[:, None, :] - dst[None], axis=2)
            best = np.argmin(gaps, axis=1)
            indices[start:start + chunk] = best
            distances[start:start + chunk] = np.take_along_axis(
                gaps, best[:, None], axis=1)[:, 0]
        return indices, distances

    order = (backend,) if backend else accelerated_backends(gpu)
    (indices, distances), _name = _try(order, run, backend)
    return indices, np.asarray(distances, dtype=np.float32)
